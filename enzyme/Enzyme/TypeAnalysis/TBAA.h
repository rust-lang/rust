//===- TBAA.h - Helpers for llvm::Type-based alias analysis   ------------===//
//
//                   Enzyme Project and The LLVM Project
// First section modified from: TypeBasedAliasAnalysis.cpp in LLVM
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
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of several utilities for understanding
// TBAA metadata and converting that metadata into corresponding TypeAnalysis
// representations.
//
//===----------------------------------------------------------------------===//
#ifndef ENZYME_TYPE_ANALYSIS_TBAA_H
#define ENZYME_TYPE_ANALYSIS_TBAA_H 1

#include "BaseType.h"
#include "ConcreteType.h"
#include "TypeTree.h"

/// isNewFormatTypeNode - Return true iff the given type node is in the new
/// size-aware format.
static bool isNewFormatTypeNode(const llvm::MDNode *N) {
  if (N->getNumOperands() < 3)
    return false;
  // In the old format the first operand is a string.
  if (!llvm::isa<llvm::MDNode>(N->getOperand(0)))
    return false;
  return true;
}

/// This is a simple wrapper around an llvm::MDNode which provides a
/// higher-level interface by hiding the details of how alias analysis
/// information is encoded in its operands.
template <typename MDNodeTy> class TBAANodeImpl {
  MDNodeTy *Node = nullptr;

public:
  TBAANodeImpl() = default;
  explicit TBAANodeImpl(MDNodeTy *N) : Node(N) {}

  /// getNode - Get the llvm::MDNode for this TBAANode.
  MDNodeTy *getNode() const { return Node; }

  /// isNewFormat - Return true iff the wrapped type node is in the new
  /// size-aware format.
  bool isNewFormat() const { return isNewFormatTypeNode(Node); }

  /// getParent - Get this TBAANode's Alias tree parent.
  TBAANodeImpl<MDNodeTy> getParent() const {
    if (isNewFormat())
      return TBAANodeImpl(llvm::cast<MDNodeTy>(Node->getOperand(0)));

    if (Node->getNumOperands() < 2)
      return TBAANodeImpl<MDNodeTy>();
    MDNodeTy *P = llvm::dyn_cast_or_null<MDNodeTy>(Node->getOperand(1));
    if (!P)
      return TBAANodeImpl<MDNodeTy>();
    // Ok, this node has a valid parent. Return it.
    return TBAANodeImpl<MDNodeTy>(P);
  }

  /// Test if this TBAANode represents a type for objects which are
  /// not modified (by any means) in the context where this
  /// AliasAnalysis is relevant.
  bool isTypeImmutable() const {
    if (Node->getNumOperands() < 3)
      return false;
    llvm::ConstantInt *CI =
        llvm::mdconst::dyn_extract<llvm::ConstantInt>(Node->getOperand(2));
    if (!CI)
      return false;
    return CI->getValue()[0];
  }
};

/// \name Specializations of \c TBAANodeImpl for const and non const qualified
/// \c MDNode.
/// @{
using TBAANode = TBAANodeImpl<const llvm::MDNode>;
using MutableTBAANode = TBAANodeImpl<llvm::MDNode>;
/// @}

/// This is a simple wrapper around an llvm::MDNode which provides a
/// higher-level interface by hiding the details of how alias analysis
/// information is encoded in its operands.
template <typename MDNodeTy> class TBAAStructTagNodeImpl {
  /// This node should be created with createTBAAAccessTag().
  MDNodeTy *Node;

public:
  explicit TBAAStructTagNodeImpl(MDNodeTy *N) : Node(N) {}

  /// Get the llvm::MDNode for this TBAAStructTagNode.
  MDNodeTy *getNode() const { return Node; }

  /// isNewFormat - Return true iff the wrapped access tag is in the new
  /// size-aware format.
  bool isNewFormat() const {
    if (Node->getNumOperands() < 4)
      return false;
    if (MDNodeTy *AccessType = getAccessType())
      if (!TBAANodeImpl<MDNodeTy>(AccessType).isNewFormat())
        return false;
    return true;
  }

  MDNodeTy *getBaseType() const {
    return llvm::dyn_cast_or_null<llvm::MDNode>(Node->getOperand(0));
  }

  MDNodeTy *getAccessType() const {
    return llvm::dyn_cast_or_null<llvm::MDNode>(Node->getOperand(1));
  }

  uint64_t getOffset() const {
    return llvm::mdconst::extract<llvm::ConstantInt>(Node->getOperand(2))
        ->getZExtValue();
  }

  uint64_t getSize() const {
    if (!isNewFormat())
      return UINT64_MAX;
    return llvm::mdconst::extract<llvm::ConstantInt>(Node->getOperand(3))
        ->getZExtValue();
  }

  /// Test if this TBAAStructTagNode represents a type for objects
  /// which are not modified (by any means) in the context where this
  /// AliasAnalysis is relevant.
  bool isTypeImmutable() const {
    unsigned OpNo = isNewFormat() ? 4 : 3;
    if (Node->getNumOperands() < OpNo + 1)
      return false;
    llvm::ConstantInt *CI =
        llvm::mdconst::dyn_extract<llvm::ConstantInt>(Node->getOperand(OpNo));
    if (!CI)
      return false;
    return CI->getValue()[0];
  }
};

/// \name Specializations of \c TBAAStructTagNodeImpl for const and non const
/// qualified \c MDNods.
/// @{
using TBAAStructTagNode = TBAAStructTagNodeImpl<const llvm::MDNode>;
using MutableTBAAStructTagNode = TBAAStructTagNodeImpl<llvm::MDNode>;
/// @}

/// This is a simple wrapper around an llvm::MDNode which provides a
/// higher-level interface by hiding the details of how alias analysis
/// information is encoded in its operands.
class TBAAStructTypeNode {
  /// This node should be created with createTBAATypeNode().
  const llvm::MDNode *Node = nullptr;

public:
  TBAAStructTypeNode() = default;
  explicit TBAAStructTypeNode(const llvm::MDNode *N) : Node(N) {}

  /// Get the llvm::MDNode for this TBAAStructTypeNode.
  const llvm::MDNode *getNode() const { return Node; }

  /// isNewFormat - Return true iff the wrapped type node is in the new
  /// size-aware format.
  bool isNewFormat() const { return isNewFormatTypeNode(Node); }

  bool operator==(const TBAAStructTypeNode &Other) const {
    return getNode() == Other.getNode();
  }

  /// getId - Return type identifier.
  llvm::Metadata *getId() const {
    return Node->getOperand(isNewFormat() ? 2 : 0);
  }

  unsigned getNumFields() const {
    unsigned FirstFieldOpNo = isNewFormat() ? 3 : 1;
    unsigned NumOpsPerField = isNewFormat() ? 3 : 2;
    return (getNode()->getNumOperands() - FirstFieldOpNo) / NumOpsPerField;
  }

  uint64_t getFieldOffset(unsigned FieldIndex) const {
    unsigned FirstFieldOpNo = isNewFormat() ? 3 : 1;
    unsigned NumOpsPerField = isNewFormat() ? 3 : 2;
    unsigned OpIndex = FirstFieldOpNo + FieldIndex * NumOpsPerField;

    uint64_t Cur =
        llvm::mdconst::extract<llvm::ConstantInt>(Node->getOperand(OpIndex + 1))
            ->getZExtValue();
    return Cur;
  }

  TBAAStructTypeNode getFieldType(unsigned FieldIndex) const {
    unsigned FirstFieldOpNo = isNewFormat() ? 3 : 1;
    unsigned NumOpsPerField = isNewFormat() ? 3 : 2;
    unsigned OpIndex = FirstFieldOpNo + FieldIndex * NumOpsPerField;
    auto *TypeNode = llvm::cast<llvm::MDNode>(getNode()->getOperand(OpIndex));
    return TBAAStructTypeNode(TypeNode);
  }

  /// Get this TBAAStructTypeNode's field in the type DAG with
  /// given offset. Update the offset to be relative to the field type.
  TBAAStructTypeNode getField(uint64_t &Offset) const {
    bool NewFormat = isNewFormat();
    if (NewFormat) {
      // New-format root and scalar type nodes have no fields.
      if (Node->getNumOperands() < 6)
        return TBAAStructTypeNode();
    } else {
      // Parent can be omitted for the root node.
      if (Node->getNumOperands() < 2)
        return TBAAStructTypeNode();

      // Fast path for a scalar type node and a struct type node with a single
      // field.
      if (Node->getNumOperands() <= 3) {
        uint64_t Cur =
            Node->getNumOperands() == 2
                ? 0
                : llvm::mdconst::extract<llvm::ConstantInt>(Node->getOperand(2))
                      ->getZExtValue();
        Offset -= Cur;
        llvm::MDNode *P =
            llvm::dyn_cast_or_null<llvm::MDNode>(Node->getOperand(1));
        if (!P)
          return TBAAStructTypeNode();
        return TBAAStructTypeNode(P);
      }
    }

    // Assume the offsets are in order. We return the previous field if
    // the current offset is bigger than the given offset.
    unsigned FirstFieldOpNo = NewFormat ? 3 : 1;
    unsigned NumOpsPerField = NewFormat ? 3 : 2;
    unsigned TheIdx = 0;
    for (unsigned Idx = FirstFieldOpNo; Idx < Node->getNumOperands();
         Idx += NumOpsPerField) {
      uint64_t Cur =
          llvm::mdconst::extract<llvm::ConstantInt>(Node->getOperand(Idx + 1))
              ->getZExtValue();
      if (Cur > Offset) {
        assert(Idx >= FirstFieldOpNo + NumOpsPerField &&
               "TBAAStructTypeNode::getField should have an offset match!");
        TheIdx = Idx - NumOpsPerField;
        break;
      }
    }
    // Move along the last field.
    if (TheIdx == 0)
      TheIdx = Node->getNumOperands() - NumOpsPerField;
    uint64_t Cur =
        llvm::mdconst::extract<llvm::ConstantInt>(Node->getOperand(TheIdx + 1))
            ->getZExtValue();
    Offset -= Cur;
    llvm::MDNode *P =
        llvm::dyn_cast_or_null<llvm::MDNode>(Node->getOperand(TheIdx));
    if (!P)
      return TBAAStructTypeNode();
    return TBAAStructTypeNode(P);
  }
};

/// Check the first operand of the tbaa tag node, if it is a llvm::MDNode, we
/// treat it as struct-path aware TBAA format, otherwise, we treat it as scalar
/// TBAA format.
static inline bool isStructPathTBAA(const llvm::MDNode *MD) {
  // Anonymous TBAA root starts with a llvm::MDNode and dragonegg uses it as
  // a TBAA tag.
  return llvm::isa<llvm::MDNode>(MD->getOperand(0)) &&
         MD->getNumOperands() >= 3;
}

static inline const llvm::MDNode *
createAccessTag(const llvm::MDNode *AccessType) {
  // If there is no access type or the access type is the root node, then
  // we don't have any useful access tag to return.
  if (!AccessType || AccessType->getNumOperands() < 2)
    return nullptr;

  llvm::Type *Int64 = llvm::IntegerType::get(AccessType->getContext(), 64);
  auto *OffsetNode =
      llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(Int64, 0));

  if (TBAAStructTypeNode(AccessType).isNewFormat()) {
    // TODO: Take access ranges into account when matching access tags and
    // fix this code to generate actual access sizes for generic tags.
    uint64_t AccessSize = UINT64_MAX;
    auto *SizeNode = llvm::ConstantAsMetadata::get(
        llvm::ConstantInt::get(Int64, AccessSize));
    llvm::Metadata *Ops[] = {const_cast<llvm::MDNode *>(AccessType),
                             const_cast<llvm::MDNode *>(AccessType), OffsetNode,
                             SizeNode};
    return llvm::MDNode::get(AccessType->getContext(), Ops);
  }

  llvm::Metadata *Ops[] = {const_cast<llvm::MDNode *>(AccessType),
                           const_cast<llvm::MDNode *>(AccessType), OffsetNode};
  return llvm::MDNode::get(AccessType->getContext(), Ops);
}

// Modified from llvm::MDNode::isTBAAVtableAccess()

static inline std::string
getAccessNameTBAA(const llvm::MDNode *M,
                  const std::set<std::string> &legalnames) {
  if (!isStructPathTBAA(M)) {
    if (M->getNumOperands() < 1)
      return "";
    if (const llvm::MDString *Tag1 =
            llvm::dyn_cast<llvm::MDString>(M->getOperand(0))) {
      return Tag1->getString().str();
    }
    return "";
  }

  // For struct-path aware TBAA, we use the access type of the tag.
  // llvm::errs() << "M: " << *M << "\n";
  TBAAStructTagNode Tag(M);
  // llvm::errs() << "AT: " << *Tag.getAccessType() << "\n";
  TBAAStructTypeNode AccessType(Tag.getAccessType());

  // llvm::errs() << "numfields: " << AccessType.getNumFields() << "\n";
  while (AccessType.getNumFields() > 0) {

    if (auto *Id = llvm::dyn_cast<llvm::MDString>(AccessType.getId())) {
      // llvm::errs() << "cur access type: " << Id->getString() << "\n";
      if (legalnames.count(Id->getString().str())) {
        return Id->getString().str();
      }
    }

    AccessType = AccessType.getFieldType(0);
    // llvm::errs() << "numfields: " << AccessType.getNumFields() << "\n";
  }

  if (auto *Id = llvm::dyn_cast<llvm::MDString>(AccessType.getId())) {
    // llvm::errs() << "access type: " << Id->getString() << "\n";
    return Id->getString().str();
  }
  return "";
}

static inline std::string
getAccessNameTBAA(llvm::Instruction *Inst,
                  const std::set<std::string> &legalnames) {
  if (const llvm::MDNode *M =
          Inst->getMetadata(llvm::LLVMContext::MD_tbaa_struct)) {
    for (unsigned i = 2; i < M->getNumOperands(); i += 3) {
      if (const llvm::MDNode *M2 =
              llvm::dyn_cast<llvm::MDNode>(M->getOperand(i))) {
        auto res = getAccessNameTBAA(M2, legalnames);
        if (res != "")
          return res;
      }
    }
  }
  if (const llvm::MDNode *M = Inst->getMetadata(llvm::LLVMContext::MD_tbaa)) {
    return getAccessNameTBAA(M, legalnames);
  }
  return "";
}

//! The following is not taken from LLVM

extern "C" {
/// Flag to print llvm::Type Analysis results as they are derived
extern llvm::cl::opt<bool> EnzymePrintType;
}

/// Derive the ConcreteType corresponding to the string TypeName
/// The llvm::Instruction I denotes the context in which this was found
static inline ConcreteType getTypeFromTBAAString(std::string TypeName,
                                                 llvm::Instruction &I) {
  if (TypeName == "long long" || TypeName == "long" || TypeName == "int" ||
      TypeName == "bool" || TypeName == "jtbaa_arraysize" ||
      TypeName == "jtbaa_arraylen") {
    if (EnzymePrintType) {
      llvm::errs() << "known tbaa " << I << " " << TypeName << "\n";
    }
    return ConcreteType(BaseType::Integer);
  } else if (TypeName == "any pointer" || TypeName == "vtable pointer" ||
             TypeName == "jtbaa_arrayptr" || TypeName == "jtbaa_tag") {
    if (EnzymePrintType) {
      llvm::errs() << "known tbaa " << I << " " << TypeName << "\n";
    }
    return ConcreteType(BaseType::Pointer);
  } else if (TypeName == "float") {
    if (EnzymePrintType)
      llvm::errs() << "known tbaa " << I << " " << TypeName << "\n";
    return llvm::Type::getFloatTy(I.getContext());
  } else if (TypeName == "double") {
    if (EnzymePrintType)
      llvm::errs() << "known tbaa " << I << " " << TypeName << "\n";
    return llvm::Type::getDoubleTy(I.getContext());
  }
  return ConcreteType(BaseType::Unknown);
}

/// Given a TBAA access node return the corresponding TypeTree
/// This includes recursively parsing the access nodes, with
/// corresponding offsets in the result
static inline TypeTree parseTBAA(TBAAStructTypeNode AccessType,
                                 llvm::Instruction &I,
                                 const llvm::DataLayout &DL) {

  if (auto *Id = llvm::dyn_cast<llvm::MDString>(AccessType.getId())) {
    auto CT = getTypeFromTBAAString(Id->getString().str(), I);
    if (CT.isKnown()) {
      return TypeTree(CT).Only(-1, &I);
    }
  }

  TypeTree Result(BaseType::Pointer);
  for (unsigned i = 0, size = AccessType.getNumFields(); i < size; ++i) {
    auto SubAccess = AccessType.getFieldType(i);
    auto Offset = AccessType.getFieldOffset(i);
    auto SubResult = parseTBAA(SubAccess, I, DL);
    Result |= SubResult.ShiftIndices(DL, /*init offset*/ 0, /*max size*/ -1,
                                     /*addOffset*/ Offset);
  }

  return Result;
}

/// Given a TBAA metadata node return the corresponding TypeTree
/// Modified from llvm::MDNode::isTBAAVtableAccess()
static inline TypeTree parseTBAA(const llvm::MDNode *M, llvm::Instruction &I,
                                 const llvm::DataLayout &DL) {
  if (!isStructPathTBAA(M)) {
    if (M->getNumOperands() < 1)
      return TypeTree();
    if (const llvm::MDString *Tag1 =
            llvm::dyn_cast<llvm::MDString>(M->getOperand(0))) {
      return TypeTree(getTypeFromTBAAString(Tag1->getString().str(), I))
          .Only(0, &I);
    }
    return TypeTree();
  }

  // For struct-path aware TBAA, we use the access type of the tag.
  TBAAStructTagNode Tag(M);
  TBAAStructTypeNode AccessType(Tag.getAccessType());
  return parseTBAA(AccessType, I, DL);
}

/// Given an llvm::Instruction, return a TypeTree representing any
/// types that can be derived from TBAA metadata attached
static inline TypeTree parseTBAA(llvm::Instruction &I,
                                 const llvm::DataLayout &DL) {
  TypeTree Result;
  if (const llvm::MDNode *M =
          I.getMetadata(llvm::LLVMContext::MD_tbaa_struct)) {
    for (unsigned i = 0, size = M->getNumOperands(); i < size; i += 3) {
      if (const llvm::MDNode *M2 =
              llvm::dyn_cast<llvm::MDNode>(M->getOperand(i + 2))) {
        auto SubResult = parseTBAA(M2, I, DL);
        auto Start = llvm::cast<llvm::ConstantInt>(
                         llvm::cast<llvm::ConstantAsMetadata>(M->getOperand(i))
                             ->getValue())
                         ->getLimitedValue();
        auto Len =
            llvm::cast<llvm::ConstantInt>(
                llvm::cast<llvm::ConstantAsMetadata>(M->getOperand(i + 1))
                    ->getValue())
                ->getLimitedValue();
        Result |=
            SubResult.ShiftIndices(DL, /*init offset*/ 0, /*max size*/ Len,
                                   /*add offset*/ Start);
      }
    }
  }
  if (const llvm::MDNode *M = I.getMetadata(llvm::LLVMContext::MD_tbaa)) {
    Result |= parseTBAA(M, I, DL);
  }
  Result |= TypeTree(BaseType::Pointer);
  return Result;
}

#endif
