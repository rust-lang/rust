/** This file contains functionality related to creation of the GenMCDriver,
 * including translating settings set by Miri.  */

#include "MiriInterface.hpp"

// CXX.rs generated headers:
#include "genmc-sys/src/lib.rs.h"

// GenMC headers:
#include "Support/Error.hpp"
#include "Support/Verbosity.hpp"
#include "Verification/InterpreterCallbacks.hpp"

// C++ headers:
#include <cstdint>

/**
 * Translate the Miri-GenMC `LogLevel` to the GenMC `VerbosityLevel`.
 * Downgrade any debug options to `Tip` if `ENABLE_GENMC_DEBUG` is not enabled.
 */
static auto to_genmc_verbosity_level(const LogLevel log_level) -> VerbosityLevel {
    switch (log_level) {
        case LogLevel::Quiet:
            return VerbosityLevel::Quiet;
        case LogLevel::Error:
            return VerbosityLevel::Error;
        case LogLevel::Warning:
            return VerbosityLevel::Warning;
        case LogLevel::Tip:
            return VerbosityLevel::Tip;
#ifdef ENABLE_GENMC_DEBUG
        case LogLevel::Debug1Revisits:
            return VerbosityLevel::Debug1;
        case LogLevel::Debug2MemoryAccesses:
            return VerbosityLevel::Debug2;
        case LogLevel::Debug3ReadsFrom:
            return VerbosityLevel::Debug3;
#else
        // Downgrade to `Tip` if the debug levels are not available.
        case LogLevel::Debug1Revisits:
        case LogLevel::Debug2MemoryAccesses:
        case LogLevel::Debug3ReadsFrom:
            return VerbosityLevel::Tip;
#endif
        default:
            WARN_ONCE(
                "unknown-log-level",
                "Unknown `LogLevel`, defaulting to `VerbosityLevel::Tip`."
            );
            return VerbosityLevel::Tip;
    }
}

/* unsafe */ void set_log_level_raw(LogLevel log_level) {
    // The `logLevel` is a static, non-atomic variable.
    // It should never be changed if `MiriGenmcShim` still exists, since any of its methods may read
    // the `logLevel`, otherwise it may cause data races.
    logLevel = to_genmc_verbosity_level(log_level);
}

/* unsafe */ auto MiriGenmcShim::create_handle(const GenmcParams& params, bool estimation_mode)
    -> std::unique_ptr<MiriGenmcShim> {
    auto conf = std::make_shared<Config>();

    // Set whether GenMC should print execution graphs after every explored/blocked execution.
    conf->printExecGraphs =
        (params.print_execution_graphs == ExecutiongraphPrinting::Explored ||
         params.print_execution_graphs == ExecutiongraphPrinting::ExploredAndBlocked);
    conf->printBlockedExecs =
        (params.print_execution_graphs == ExecutiongraphPrinting::Blocked ||
         params.print_execution_graphs == ExecutiongraphPrinting::ExploredAndBlocked);

    // `1024` is the default value that GenMC uses.
    // If any thread has at least this many events, a warning/tip will be printed.
    //
    // Miri produces a lot more events than GenMC, so the graph size warning triggers on almost
    // all programs. The current value is large enough so the warning is not be triggered by any
    // reasonable programs.
    // FIXME(genmc): The emitted warning mentions features not supported by Miri ('--unroll'
    // parameter).
    // FIXME(genmc): A more appropriate limit should be chosen once the warning is useful for
    // Miri.
    conf->warnOnGraphSize = 1024 * 1024;

    // We only support the `RC11` memory model for Rust, and `SC` when weak memory emulation is
    // disabled.
    conf->model = params.disable_weak_memory_emulation ? ModelType::SC : ModelType::RC11;

    // This prints the seed that GenMC picks for randomized scheduling during estimation mode.
    conf->printRandomScheduleSeed = params.print_random_schedule_seed;

    // FIXME(genmc): supporting IPR requires annotations for `assume` and `spinloops`.
    conf->ipr = false;
    // FIXME(genmc): supporting BAM requires `Barrier` support + detecting whether return value
    // of barriers are used.
    conf->disableBAM = true;

    // Instruction caching could help speed up verification by filling the graph from cache, if
    // the list of values read by all load events in a thread have been seen before. Combined
    // with not replaying completed threads, this can also reducing the amount of Mir
    // interpretation required by Miri. With the current setup, this would be incorrect, since
    // Miri doesn't give GenMC the actual values read by non-atomic reads.
    conf->instructionCaching = false;
    // Many of Miri's checks work under the assumption that threads are only executed in an
    // order that could actually happen during a normal execution. Formally, this means that
    // replaying an execution needs to respect the po-rf-relation of the executiongraph (po ==
    // program-order, rf == reads-from). This means, any event in the graph, when replayed, must
    // happen after any events that happen before it in the same graph according to the program
    // code, and all (non-atomic) reads must happen after the write event they read from.
    //
    // Not replaying completed threads means any read event from that thread never happens in
    // Miri's memory, so this would only work if there are never any non-atomic reads from any
    // value written by the skipped thread.
    conf->replayCompletedThreads = true;

    // FIXME(genmc): implement symmetry reduction.
    ERROR_ON(
        params.do_symmetry_reduction,
        "Symmetry reduction is currently unsupported in GenMC mode."
    );
    conf->symmetryReduction = params.do_symmetry_reduction;

    // Set the scheduling policy. GenMC uses `WFR` for estimation mode.
    // For normal verification, `WF` has the best performance and is the GenMC default.
    // Other scheduling policies are used by GenMC for testing and for modes currently
    // unsupported with Miri such as bounding, which uses LTR.
    conf->schedulePolicy = estimation_mode ? SchedulePolicy::WFR : SchedulePolicy::WF;

    // Set the min and max number of executions tested in estimation mode.
    conf->estimationMin = 10; // default taken from GenMC
    conf->estimationMax = params.estimation_max;
    // Deviation threshold % under which estimation is deemed good enough.
    conf->sdThreshold = 10; // default taken from GenMC
    // Set the mode used for this driver, either estimation or verification.
    const auto mode = estimation_mode ? GenMCDriver::Mode(GenMCDriver::EstimationMode {})
                                      : GenMCDriver::Mode(GenMCDriver::VerificationMode {});

    // Running Miri-GenMC without race detection is not supported.
    // Disabling this option also changes the behavior of the replay scheduler to only schedule
    // at atomic operations, which is required with Miri. This happens because Miri can generate
    // multiple GenMC events for a single MIR terminator. Without this option, the scheduler
    // might incorrectly schedule an atomic MIR terminator because the first event it creates is
    // a non-atomic (e.g., `StorageLive`).
    conf->disableRaceDetection = false;

    // Miri can already check for unfreed memory. Also, GenMC cannot distinguish between memory
    // that is allowed to leak and memory that is not.
    conf->warnUnfreedMemory = false;

    // FIXME(genmc,error handling): This function currently exits on error, but will return an
    // error value in the future. The return value should be checked once this change is made.
    checkConfig(*conf);

    // Create the actual driver and Miri-GenMC communication shim.
    auto driver = std::make_unique<MiriGenmcShim>(std::move(conf), mode);

    // FIXME(genmc,HACK): Until a proper solution is implemented in GenMC, these callbacks will
    // allow Miri to return information about global allocations and override uninitialized memory
    // checks for non-atomic loads (Miri handles those without GenMC, so the error would be wrong).
    auto interpreter_callbacks = InterpreterCallbacks {
        // Miri already ensures that memory accesses are valid, so this check doesn't matter.
        // We check that the address is static, but skip checking if it is part of an actual
        // allocation.
        .isStaticallyAllocated = [](SAddr addr) { return addr.isStatic(); },
        // FIXME(genmc,error reporting): Once a proper a proper API for passing such information is
        // implemented in GenMC, Miri should use it to improve the produced error messages.
        .getStaticName = [](SAddr addr) { return "[UNKNOWN STATIC]"; },
        // This function is called to get the initial value stored at the given address.
        //
        // From a Miri perspective, this API doesn't work very well: most memory starts out
        // "uninitialized";
        // only statics have an initial value. And their initial value is just a sequence of bytes,
        // but GenMC expect this to be already split into separate atomic variables. So we return a
        // dummy value.
        // This value should never be visible to the interpreted program.
        // GenMC does not understand uninitialized memory the same way Miri does, which may cause
        // this function to be called. The returned value can be visible to Miri or the user:
        // - Printing the execution graph may contain this value in place of uninitialized values.
        //   FIXME(genmc): NOTE: printing the execution graph is not yet implemented.
        // - Non-atomic loads may return this value, but Miri ignores values of non-atomic loads.
        // - Atomic loads will *not* see this value once mixed atomic-non-atomic support is added.
        //   Currently, atomic loads can see this value, unless initialized by an *atomic* store.
        //   FIXME(genmc): update this comment once mixed atomic-non-atomic support is added.
        //
        // FIXME(genmc): implement proper support for uninitialized memory in GenMC.
        // Ideally, the initial value getter would return an `optional<SVal>`, since the memory
        // location may be uninitialized.
        .initValGetter = [](const AAccess& a) { return SVal(0xDEAD); },
        // Miri serves non-atomic loads from its own memory and these GenMC checks are wrong in that
        // case. This should no longer be required with proper mixed-size access support.
        .skipUninitLoadChecks = [](const MemAccessLabel* access_label
                                ) { return access_label->getOrdering() == MemOrdering::NotAtomic; },
    };
    driver->setInterpCallbacks(std::move(interpreter_callbacks));

    return driver;
}
