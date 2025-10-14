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

/**** Blocking instructions ****/

void MiriGenmcShim::handle_assume_block(ThreadId thread_id, AssumeType assume_type) {
    BUG_ON(getExec().getGraph().isThreadBlocked(thread_id));
    GenMCDriver::handleAssume(inc_pos(thread_id), assume_type);
}

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
        GenmcScalarExt::try_to_sval(old_val),
        ord,
        SAddr(address),
        ASize(size),
        type
    );

    if (const auto* err = std::get_if<VerificationError>(&ret))
        return LoadResultExt::from_error(format_error(*err));
    const auto* ret_val = std::get_if<SVal>(&ret);
    // FIXME(genmc): handle `HandleResult::{Invalid, Reset}` return values.
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
        GenmcScalarExt::try_to_sval(old_val),
        ord,
        SAddr(address),
        ASize(size),
        /* type */ AType::Unsigned, // `type` is only used for printing.
        GenmcScalarExt::to_sval(value),
        EventDeps()
    );

    if (const auto* err = std::get_if<VerificationError>(&ret))
        return StoreResultExt::from_error(format_error(*err));

    const bool* is_coherence_order_maximal_write = std::get_if<bool>(&ret);
    // FIXME(genmc): handle `HandleResult::{Invalid, Reset}` return values.
    ERROR_ON(
        nullptr == is_coherence_order_maximal_write,
        "Unimplemented: Store returned unexpected result."
    );
    return StoreResultExt::ok(*is_coherence_order_maximal_write);
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
        GenmcScalarExt::try_to_sval(old_val),
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
    // FIXME(genmc): handle `HandleResult::{Invalid, Reset}` return values.
    if (nullptr == ret_val) {
        ERROR("Unimplemented: read-modify-write returned unexpected result.");
    }
    const auto read_old_val = *ret_val;
    const auto new_value =
        executeRMWBinOp(read_old_val, GenmcScalarExt::to_sval(rhs_value), size, rmw_op);

    const auto storePos = inc_pos(thread_id);
    const auto store_ret = GenMCDriver::handleStore<EventLabel::EventLabelKind::FaiWrite>(
        storePos,
        GenmcScalarExt::try_to_sval(old_val),
        ordering,
        SAddr(address),
        ASize(size),
        AType::Unsigned, // The type is only used for printing.
        new_value
    );
    if (const auto* err = std::get_if<VerificationError>(&store_ret))
        return ReadModifyWriteResultExt::from_error(format_error(*err));

    const bool* is_coherence_order_maximal_write = std::get_if<bool>(&store_ret);
    // FIXME(genmc): handle `HandleResult::{Invalid, Reset}` return values.
    ERROR_ON(
        nullptr == is_coherence_order_maximal_write,
        "Unimplemented: RMW store returned unexpected result."
    );
    return ReadModifyWriteResultExt::ok(
        /* old_value: */ read_old_val,
        new_value,
        *is_coherence_order_maximal_write
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
        GenmcScalarExt::try_to_sval(old_val),
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
    // FIXME(genmc): handle `HandleResult::{Invalid, Reset}` return values.
    ERROR_ON(nullptr == ret_val, "Unimplemented: load returned unexpected result.");
    const auto read_old_val = *ret_val;
    if (read_old_val != expectedVal)
        return CompareExchangeResultExt::failure(read_old_val);

    // FIXME(GenMC): Add support for modelling spurious failures.

    const auto storePos = inc_pos(thread_id);
    const auto store_ret = GenMCDriver::handleStore<EventLabel::EventLabelKind::CasWrite>(
        storePos,
        GenmcScalarExt::try_to_sval(old_val),
        success_ordering,
        SAddr(address),
        ASize(size),
        AType::Unsigned, // The type is only used for printing.
        new_val
    );
    if (const auto* err = std::get_if<VerificationError>(&store_ret))
        return CompareExchangeResultExt::from_error(format_error(*err));
    const bool* is_coherence_order_maximal_write = std::get_if<bool>(&store_ret);
    // FIXME(genmc): handle `HandleResult::{Invalid, Reset}` return values.
    ERROR_ON(
        nullptr == is_coherence_order_maximal_write,
        "Unimplemented: compare-exchange store returned unexpected result."
    );
    return CompareExchangeResultExt::success(read_old_val, *is_coherence_order_maximal_write);
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

auto MiriGenmcShim::handle_free(ThreadId thread_id, uint64_t address) -> bool {
    const auto pos = inc_pos(thread_id);
    GenMCDriver::handleFree(pos, SAddr(address), EventDeps());
    // FIXME(genmc): use returned error from `handleFree` once implemented in GenMC.
    return getResult().status.has_value();
}
