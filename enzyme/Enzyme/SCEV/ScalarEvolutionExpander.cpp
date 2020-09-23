#include "SCEV/ScalarEvolutionExpander.h"

#include "llvm/Config/llvm-config.h"

#if LLVM_VERSION_MAJOR >= 12
#include "ScalarEvolutionExpander12.cpp"
#elif LLVM_VERSION_MAJOR >= 11
#include "ScalarEvolutionExpander11.cpp"
#elif LLVM_VERSION_MAJOR >= 9
#include "ScalarEvolutionExpander9.cpp"
#elif LLVM_VERSION_MAJOR >= 7
#include "ScalarEvolutionExpander8.cpp"
#else
#include "ScalarEvolutionExpander6.cpp"
#endif
