#ifndef GENMC_MIRI_INTERFACE_HPP
#define GENMC_MIRI_INTERFACE_HPP

#include "rust/cxx.h"

#include "config.h"

#include "Config/Config.hpp"
#include "Verification/GenMCDriver.hpp"

#include <iostream>

/**** Types available to Miri ****/

// Config struct defined on the Rust side and translated to C++ by cxx.rs:
struct GenmcParams;

struct MiriGenMCShim : private GenMCDriver
{

public:
	MiriGenMCShim(std::shared_ptr<const Config> conf, Mode mode /* = VerificationMode{} */)
		: GenMCDriver(std::move(conf), nullptr, mode)
	{
		std::cerr << "C++: GenMC handle created!" << std::endl;
	}

	virtual ~MiriGenMCShim()
	{
		std::cerr << "C++: GenMC handle destroyed!" << std::endl;
	}

	static std::unique_ptr<MiriGenMCShim> createHandle(const GenmcParams &config);
};

/**** Functions available to Miri ****/

// NOTE: CXX doesn't support exposing static methods to Rust currently, so we expose this function instead.
static inline auto createGenmcHandle(const GenmcParams &config) -> std::unique_ptr<MiriGenMCShim>
{
	return MiriGenMCShim::createHandle(config);
}

#endif /* GENMC_MIRI_INTERFACE_HPP */
