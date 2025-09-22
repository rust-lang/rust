/** This file contains functionality related to handling events encountered
 * during an execution, such as loads, stores or memory (de)allocation. */

#include "MiriInterface.hpp"

// CXX.rs generated headers:
#include "genmc-sys/src/lib.rs.h"

// GenMC headers:
#include "ADT/value_ptr.hpp"
#include "ExecutionGraph/EventLabel.hpp"
#include "ExecutionGraph/LoadAnnotation.hpp"
#include "Runtime/InterpreterEnumAPI.hpp"
#include "Static/ModuleID.hpp"
#include "Support/ASize.hpp"
#include "Support/Error.hpp"
#include "Support/Logger.hpp"
#include "Support/MemAccess.hpp"
#include "Support/RMWOps.hpp"
#include "Support/SAddr.hpp"
#include "Support/SVal.hpp"
#include "Support/ThreadInfo.hpp"
#include "Support/Verbosity.hpp"
#include "Verification/GenMCDriver.hpp"
#include "Verification/MemoryModel.hpp"

// C++ headers:
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

/**** Memory access handling ****/

[[nodiscard]] auto MiriGenmcShim::handle_load(
    ThreadId thread_id,
    uint64_t address,
    uint64_t size,
    MemOrdering ord,
    GenmcScalar old_val
) -> LoadResult {
    // `type` is only used for printing.
    const auto type = AType::Unsigned;
    const auto ret = handle_load_reset_if_none<EventLabel::EventLabelKind::Read>(
        thread_id,
        ord,
        SAddr(address),
        ASize(size),
        type
    );

    if (const auto* err = std::get_if<VerificationError>(&ret))
        return LoadResultExt::from_error(format_error(*err));
    const auto* ret_val = std::get_if<SVal>(&ret);
    if (ret_val == nullptr)
        ERROR("Unimplemented: load returned unexpected result.");
    return LoadResultExt::from_value(*ret_val);
}

[[nodiscard]] auto MiriGenmcShim::handle_store(
    ThreadId thread_id,
    uint64_t address,
    uint64_t size,
    GenmcScalar value,
    GenmcScalar old_val,
    MemOrdering ord
) -> StoreResult {
    const auto pos = inc_pos(thread_id);
    const auto ret = GenMCDriver::handleStore<EventLabel::EventLabelKind::Write>(
        pos,
        ord,
        SAddr(address),
        ASize(size),
        /* type */ AType::Unsigned, // `type` is only used for printing.
        GenmcScalarExt::to_sval(value),
        EventDeps()
    );

    if (const auto* err = std::get_if<VerificationError>(&ret))
        return StoreResultExt::from_error(format_error(*err));
    if (!std::holds_alternative<std::monostate>(ret))
        ERROR("store returned unexpected result");

    // FIXME(genmc,mixed-accesses): Use the value that GenMC returns from handleStore (once
    // available).
    const auto& g = getExec().getGraph();
    return StoreResultExt::ok(
        /* is_coherence_order_maximal_write */ g.co_max(SAddr(address))->getPos() == pos
    );
}

void MiriGenmcShim::handle_fence(ThreadId thread_id, MemOrdering ord) {
    const auto pos = inc_pos(thread_id);
    GenMCDriver::handleFence(pos, ord, EventDeps());
}

[[nodiscard]] auto MiriGenmcShim::handle_read_modify_write(
    ThreadId thread_id,
    uint64_t address,
    uint64_t size,
    RMWBinOp rmw_op,
    MemOrdering ordering,
    GenmcScalar rhs_value,
    GenmcScalar old_val
) -> ReadModifyWriteResult {
    // NOTE: Both the store and load events should get the same `ordering`, it should not be split
    // into a load and a store component. This means we can have for example `AcqRel` loads and
    // stores, but this is intended for RMW operations.

    // Somewhat confusingly, the GenMC term for RMW read/write labels is
    // `FaiRead` and `FaiWrite`.
    const auto load_ret = handle_load_reset_if_none<EventLabel::EventLabelKind::FaiRead>(
        thread_id,
        ordering,
        SAddr(address),
        ASize(size),
        AType::Unsigned, // The type is only used for printing.
        rmw_op,
        GenmcScalarExt::to_sval(rhs_value),
        EventDeps()
    );
    if (const auto* err = std::get_if<VerificationError>(&load_ret))
        return ReadModifyWriteResultExt::from_error(format_error(*err));

    const auto* ret_val = std::get_if<SVal>(&load_ret);
    if (nullptr == ret_val) {
        ERROR("Unimplemented: read-modify-write returned unexpected result.");
    }
    const auto read_old_val = *ret_val;
    const auto new_value =
        executeRMWBinOp(read_old_val, GenmcScalarExt::to_sval(rhs_value), size, rmw_op);

    const auto storePos = inc_pos(thread_id);
    const auto store_ret = GenMCDriver::handleStore<EventLabel::EventLabelKind::FaiWrite>(
        storePos,
        ordering,
        SAddr(address),
        ASize(size),
        AType::Unsigned, // The type is only used for printing.
        new_value
    );
    if (const auto* err = std::get_if<VerificationError>(&store_ret))
        return ReadModifyWriteResultExt::from_error(format_error(*err));

    const auto* store_ret_val = std::get_if<std::monostate>(&store_ret);
    ERROR_ON(nullptr == store_ret_val, "Unimplemented: RMW store returned unexpected result.");

    // FIXME(genmc,mixed-accesses): Use the value that GenMC returns from handleStore (once
    // available).
    const auto& g = getExec().getGraph();
    return ReadModifyWriteResultExt::ok(
        /* old_value: */ read_old_val,
        new_value,
        /* is_coherence_order_maximal_write */ g.co_max(SAddr(address))->getPos() == storePos
    );
}

[[nodiscard]] auto MiriGenmcShim::handle_compare_exchange(
    ThreadId thread_id,
    uint64_t address,
    uint64_t size,
    GenmcScalar expected_value,
    GenmcScalar new_value,
    GenmcScalar old_val,
    MemOrdering success_ordering,
    MemOrdering fail_load_ordering,
    bool can_fail_spuriously
) -> CompareExchangeResult {
    // NOTE: Both the store and load events should get the same `ordering`, it should not be split
    // into a load and a store component. This means we can have for example `AcqRel` loads and
    // stores, but this is intended for CAS operations.

    // FIXME(GenMC): properly handle failure memory ordering.

    auto expectedVal = GenmcScalarExt::to_sval(expected_value);
    auto new_val = GenmcScalarExt::to_sval(new_value);

    const auto load_ret = handle_load_reset_if_none<EventLabel::EventLabelKind::CasRead>(
        thread_id,
        success_ordering,
        SAddr(address),
        ASize(size),
        AType::Unsigned, // The type is only used for printing.
        expectedVal,
        new_val
    );
    if (const auto* err = std::get_if<VerificationError>(&load_ret))
        return CompareExchangeResultExt::from_error(format_error(*err));
    const auto* ret_val = std::get_if<SVal>(&load_ret);
    ERROR_ON(nullptr == ret_val, "Unimplemented: load returned unexpected result.");
    const auto read_old_val = *ret_val;
    if (read_old_val != expectedVal)
        return CompareExchangeResultExt::failure(read_old_val);

    // FIXME(GenMC): Add support for modelling spurious failures.

    const auto storePos = inc_pos(thread_id);
    const auto store_ret = GenMCDriver::handleStore<EventLabel::EventLabelKind::CasWrite>(
        storePos,
        success_ordering,
        SAddr(address),
        ASize(size),
        AType::Unsigned, // The type is only used for printing.
        new_val
    );
    if (const auto* err = std::get_if<VerificationError>(&store_ret))
        return CompareExchangeResultExt::from_error(format_error(*err));
    const auto* store_ret_val = std::get_if<std::monostate>(&store_ret);
    ERROR_ON(
        nullptr == store_ret_val,
        "Unimplemented: compare-exchange store returned unexpected result."
    );

    // FIXME(genmc,mixed-accesses): Use the value that GenMC returns from handleStore (once
    // available).
    const auto& g = getExec().getGraph();
    return CompareExchangeResultExt::success(
        read_old_val,
        /* is_coherence_order_maximal_write */ g.co_max(SAddr(address))->getPos() == storePos
    );
}

/**** Memory (de)allocation ****/

auto MiriGenmcShim::handle_malloc(ThreadId thread_id, uint64_t size, uint64_t alignment)
    -> uint64_t {
    const auto pos = inc_pos(thread_id);

    // These are only used for printing and features Miri-GenMC doesn't support (yet).
    const auto storage_duration = StorageDuration::SD_Heap;
    // Volatile, as opposed to "persistent" (i.e., non-volatile memory that persists over reboots)
    const auto storage_type = StorageType::ST_Volatile;
    const auto address_space = AddressSpace::AS_User;

    const SVal ret_val = GenMCDriver::handleMalloc(
        pos,
        size,
        alignment,
        storage_duration,
        storage_type,
        address_space,
        EventDeps()
    );
    return ret_val.get();
}

void MiriGenmcShim::handle_free(ThreadId thread_id, uint64_t address) {
    const auto pos = inc_pos(thread_id);
    GenMCDriver::handleFree(pos, SAddr(address), EventDeps());
    // FIXME(genmc): add error handling once GenMC returns errors from `handleFree`
}
