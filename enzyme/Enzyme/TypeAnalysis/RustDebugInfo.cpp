//===- RustDebugInfo.cpp - Implementaion of Rust Debug Info Parser   ---===//
//
//                             Enzyme Project
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// If using this code in an academic setting, please cite the following:
// @incollection{enzymeNeurips,
// title = {Instead of Rewriting Foreign Code for Machine Learning,
//          Automatically Synthesize Fast Gradients},
// author = {Moses, William S. and Churavy, Valentin},
// booktitle = {Advances in Neural Information Processing Systems 33},
// year = {2020},
// note = {To appear in},
// }
//
//===-------------------------------------------------------------------===//
//
// This file implement the Rust debug info parsing function. It will get the
// description of types from debug info of an instruction and pass it to
// concrete functions according to the kind of a description and construct
// the type tree recursively.
//
//===-------------------------------------------------------------------===//
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/Support/CommandLine.h"

#include "RustDebugInfo.h"

TypeTree parseDIType(DIType &Type, Instruction &I, DataLayout &DL);

TypeTree parseDIType(DIBasicType &Type, Instruction &I, DataLayout &DL) {
  std::string TypeName = Type.getName().str();
  TypeTree Result;
  if (TypeName == "f64") {
    Result = TypeTree(Type::getDoubleTy(I.getContext())).Only(0);
  } else if (TypeName == "f32") {
    Result = TypeTree(Type::getFloatTy(I.getContext())).Only(0);
  } else if (TypeName == "i8" || TypeName == "i16" || TypeName == "i32" ||
             TypeName == "i64" || TypeName == "isize" || TypeName == "u8" ||
             TypeName == "u16" || TypeName == "u32" || TypeName == "u64" ||
             TypeName == "usize" || TypeName == "i128" || TypeName == "u128") {
    Result = TypeTree(ConcreteType(BaseType::Integer)).Only(0);
  } else {
    Result = TypeTree(ConcreteType(BaseType::Unknown)).Only(0);
  }
  return Result;
}

TypeTree parseDIType(DICompositeType &Type, Instruction &I, DataLayout &DL) {
  TypeTree Result;
  if (Type.getTag() == dwarf::DW_TAG_array_type) {
#if LLVM_VERSION_MAJOR >= 9
    DIType *SubType = Type.getBaseType();
#else
    DIType *SubType = Type.getBaseType().resolve();
#endif
    TypeTree SubTT = parseDIType(*SubType, I, DL);
    size_t Align = Type.getAlignInBytes();
    size_t SubSize = SubType->getSizeInBits() / 8;
    size_t Size = Type.getSizeInBits() / 8;
    DINodeArray Subranges = Type.getElements();
    size_t pos = 0;
    for (auto r : Subranges) {
      DISubrange *Subrange = dyn_cast<DISubrange>(r);
      if (auto Count = Subrange->getCount().get<ConstantInt *>()) {
        int64_t count = Count->getSExtValue();
        if (count == -1) {
          break;
        }
        for (int64_t i = 0; i < count; i++) {
          Result |= SubTT.ShiftIndices(DL, 0, Size, pos);
          size_t tmp = pos + SubSize;
          if (tmp % Align != 0) {
            pos = (tmp / Align + 1) * Align;
          } else {
            pos = tmp;
          }
        }
      } else {
        assert(0 && "There shouldn't be non-constant-size arrays in Rust");
      }
    }
    return Result;
  } else if (Type.getTag() == dwarf::DW_TAG_structure_type ||
             Type.getTag() == dwarf::DW_TAG_union_type) {
    DINodeArray Elements = Type.getElements();
    size_t Size = Type.getSizeInBits() / 8;
    bool firstSubTT = true;
    for (auto e : Elements) {
      DIType *SubType = dyn_cast<DIDerivedType>(e);
      assert(SubType->getTag() == dwarf::DW_TAG_member);
      TypeTree SubTT = parseDIType(*SubType, I, DL);
      size_t Offset = SubType->getOffsetInBits() / 8;
      SubTT = SubTT.ShiftIndices(DL, 0, Size, Offset);
      if (Type.getTag() == dwarf::DW_TAG_structure_type) {
        Result |= SubTT;
      } else {
        if (firstSubTT) {
          Result = SubTT;
        } else {
          Result &= SubTT;
        }
      }
      if (firstSubTT) {
        firstSubTT = !firstSubTT;
      }
    }
    return Result;
  } else {
    assert(0 && "Composite types other than arrays, structs and unions are not "
                "supported by Rust debug info parser");
  }
}

TypeTree parseDIType(DIDerivedType &Type, Instruction &I, DataLayout &DL) {
  if (Type.getTag() == dwarf::DW_TAG_pointer_type) {
    TypeTree Result(BaseType::Pointer);
#if LLVM_VERSION_MAJOR >= 9
    DIType *SubType = Type.getBaseType();
#else
    DIType *SubType = Type.getBaseType().resolve();
#endif
    TypeTree SubTT = parseDIType(*SubType, I, DL);
    if (isa<DIBasicType>(SubType)) {
      Result |= SubTT.ShiftIndices(DL, 0, 1, -1);
    } else {
      Result |= SubTT;
    }
    return Result.Only(0);
  } else if (Type.getTag() == dwarf::DW_TAG_member) {
#if LLVM_VERSION_MAJOR >= 9
    DIType *SubType = Type.getBaseType();
#else
    DIType *SubType = Type.getBaseType().resolve();
#endif
    TypeTree Result = parseDIType(*SubType, I, DL);
    return Result;
  } else {
    assert(0 && "Derived types other than pointers and members are not "
                "supported by Rust debug info parser");
  }
}

TypeTree parseDIType(DIType &Type, Instruction &I, DataLayout &DL) {
  if (Type.getSizeInBits() == 0) {
    return TypeTree();
  }

  if (auto BT = dyn_cast<DIBasicType>(&Type)) {
    return parseDIType(*BT, I, DL);
  } else if (auto CT = dyn_cast<DICompositeType>(&Type)) {
    return parseDIType(*CT, I, DL);
  } else if (auto DT = dyn_cast<DIDerivedType>(&Type)) {
    return parseDIType(*DT, I, DL);
  } else {
    assert(0 && "Types other than floating-points, integers, arrays, pointers, "
                "slices, and structs are not supported by debug info parser");
  }
}

bool isU8PointerType(DIType &type) {
  if (type.getTag() == dwarf::DW_TAG_pointer_type) {
    auto PTy = dyn_cast<DIDerivedType>(&type);
#if LLVM_VERSION_MAJOR >= 9
    DIType *SubType = PTy->getBaseType();
#else
    DIType *SubType = PTy->getBaseType().resolve();
#endif
    if (auto BTy = dyn_cast<DIBasicType>(SubType)) {
      std::string name = BTy->getName().str();
      if (name == "u8") {
        return true;
      }
    }
  }
  return false;
}

TypeTree parseDIType(DbgDeclareInst &I, DataLayout &DL) {
#if LLVM_VERSION_MAJOR >= 9
  DIType *type = I.getVariable()->getType();
#else
  DIType *type = I.getVariable()->getType().resolve();
#endif

  // If the type is *u8, do nothing, since the underlying type of data pointed
  // by a *u8 can be anything
  if (isU8PointerType(*type)) {
    return TypeTree();
  }
  TypeTree Result = parseDIType(*type, I, DL);
  return Result;
}
