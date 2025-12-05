#ifndef GENMC_RESULT_HANDLING_HPP
#define GENMC_RESULT_HANDLING_HPP

// CXX.rs generated headers:
#include "rust/cxx.h"

// GenMC headers:
#include "Verification/VerificationError.hpp"

#include <format>
#include <memory>
#include <sstream>
#include <string>

/** Information about an error, formatted as a string to avoid having to share an error enum and
 * printing functionality with the Rust side. */
static auto format_error(VerificationError err) -> std::unique_ptr<std::string> {
    std::stringstream s;
    s << std::format("{}", err);
    return std::make_unique<std::string>(s.str());
}

#endif /* GENMC_RESULT_HANDLING_HPP */
