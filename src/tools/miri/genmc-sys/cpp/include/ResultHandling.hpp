#ifndef GENMC_RESULT_HANDLING_HPP
#define GENMC_RESULT_HANDLING_HPP

// CXX.rs generated headers:
#include "rust/cxx.h"

// GenMC headers:
#include "Verification/VerificationError.hpp"

#include <string>

/** Information about an error, formatted as a string to avoid having to share an error enum and
 * printing functionality with the Rust side. */
static auto format_error(VerificationError err) -> std::unique_ptr<std::string> {
    auto buf = std::string();
    auto s = llvm::raw_string_ostream(buf);
    s << err;
    return std::make_unique<std::string>(s.str());
}

#endif /* GENMC_RESULT_HANDLING_HPP */
