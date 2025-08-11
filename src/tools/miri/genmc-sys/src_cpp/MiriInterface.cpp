#include "MiriInterface.hpp"

#include "genmc-sys/src/lib.rs.h"

auto MiriGenMCShim::createHandle(const GenmcParams &config)
	-> std::unique_ptr<MiriGenMCShim>
{
	auto conf = std::make_shared<Config>();

	// Miri needs all threads to be replayed, even fully completed ones.
	conf->replayCompletedThreads = true;

	// We only support the RC11 memory model for Rust.
	conf->model = ModelType::RC11;

	conf->printRandomScheduleSeed = config.print_random_schedule_seed;

	// FIXME(genmc): disable any options we don't support currently:
	conf->ipr = false;
	conf->disableBAM = true;
	conf->instructionCaching = false;

	ERROR_ON(config.do_symmetry_reduction, "Symmetry reduction is currently unsupported in GenMC mode.");
	conf->symmetryReduction = config.do_symmetry_reduction;

	// FIXME(genmc): Should there be a way to change this option from Miri?
	conf->schedulePolicy = SchedulePolicy::WF;

	// FIXME(genmc): implement estimation mode:
	conf->estimate = false;
	conf->estimationMax = 1000;
	const auto mode = conf->estimate ? GenMCDriver::Mode(GenMCDriver::EstimationMode{})
									  : GenMCDriver::Mode(GenMCDriver::VerificationMode{});

	// Running Miri-GenMC without race detection is not supported.
	// Disabling this option also changes the behavior of the replay scheduler to only schedule at atomic operations, which is required with Miri.
	// This happens because Miri can generate multiple GenMC events for a single MIR terminator. Without this option,
	// the scheduler might incorrectly schedule an atomic MIR terminator because the first event it creates is a non-atomic (e.g., `StorageLive`).
	conf->disableRaceDetection = false;

	// Miri can already check for unfreed memory. Also, GenMC cannot distinguish between memory
	// that is allowed to leak and memory that is not.
	conf->warnUnfreedMemory = false;

	// FIXME(genmc): check config:
	// checkConfigOptions(*conf);

	auto driver = std::make_unique<MiriGenMCShim>(std::move(conf), mode);
	return driver;
}
