//===- ScalarEvolutionExpander.cpp - Scalar Evolution Analysis ------------===//
//
//                             Enzyme Project
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// If using this code in an academic setting, please cite the following:
// @misc{enzymeGithub,
//  author = {William S. Moses and Valentin Churavy},
//  title = {Enzyme: High Performance Automatic Differentiation of LLVM},
//  year = {2020},
//  howpublished = {\url{https://github.com/wsmoses/Enzyme}},
//  note = {commit xxxxxxx}
// }
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of the scalar evolution expander
// with modifications necessary to allow usage in LLVM plugins.
//
//===----------------------------------------------------------------------===//

#include "SCEV/ScalarEvolutionExpander.h"

#include "llvm/Config/llvm-config.h"

#if LLVM_VERSION_MAJOR >= 12
#include "ScalarEvolutionExpander12.cpp"
#elif LLVM_VERSION_MAJOR >= 11
#include "ScalarEvolutionExpander11.cpp"
#elif LLVM_VERSION_MAJOR >= 9
#include "ScalarEvolutionExpander9.cpp"
#elif LLVM_VERSION_MAJOR >= 8
#include "ScalarEvolutionExpander8.cpp"
#elif LLVM_VERSION_MAJOR >= 7
#include "ScalarEvolutionExpander7.cpp"
#else
#include "ScalarEvolutionExpander6.cpp"
#endif
