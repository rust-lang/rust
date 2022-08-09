; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -adce -correlated-propagation -simplifycfg -S | FileCheck %s

; ModuleID = 'ld-temp.o'
source_filename = "ld-temp.o"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%"struct.std::_Rb_tree_node_base" = type { i32, %"struct.std::_Rb_tree_node_base"*, %"struct.std::_Rb_tree_node_base"*, %"struct.std::_Rb_tree_node_base"* }

@enzyme_dup = external dso_local local_unnamed_addr global i32, align 4
@enzyme_const = external dso_local local_unnamed_addr global i32, align 4

declare dso_local i32 @_Z17__enzyme_autodiff(...)

; Function Attrs: mustprogress nofree nounwind readonly willreturn
declare dso_local %"struct.std::_Rb_tree_node_base"* @_ZSt18_Rb_tree_incrementPKSt18_Rb_tree_node_base(%"struct.std::_Rb_tree_node_base"*) local_unnamed_addr

; Function Attrs: mustprogress nofree nounwind readonly willreturn
declare dso_local %"struct.std::_Rb_tree_node_base"* @_ZSt18_Rb_tree_decrementPKSt18_Rb_tree_node_base(%"struct.std::_Rb_tree_node_base"*) local_unnamed_addr
  
define dso_local i32 @callable(%"struct.std::_Rb_tree_node_base"* %i0) {
  %i1 = call %"struct.std::_Rb_tree_node_base"* @_ZSt18_Rb_tree_incrementPKSt18_Rb_tree_node_base(%"struct.std::_Rb_tree_node_base"* %i0)
  %i2 = call %"struct.std::_Rb_tree_node_base"* @_ZSt18_Rb_tree_decrementPKSt18_Rb_tree_node_base(%"struct.std::_Rb_tree_node_base"* %i1)
  ret i32 0 
}

define dso_local i32 @main(%"struct.std::_Rb_tree_node_base"* %i0, %"struct.std::_Rb_tree_node_base"* %shaddow) {
  %id = load i32, i32* @enzyme_dup, align 4
  %i1 = call i32 (...) @_Z17__enzyme_autodiff(i8* bitcast (i32 (%"struct.std::_Rb_tree_node_base"*)* @callable to i8*), i32 %id, %"struct.std::_Rb_tree_node_base"* nonnull align 8 dereferenceable(24) %i0, %"struct.std::_Rb_tree_node_base"* nonnull align 8 dereferenceable(24) %shaddow)
  ret i32 0
}

; CHECK: define internal void @diffecallable(%"struct.std::_Rb_tree_node_base"* %i0, %"struct.std::_Rb_tree_node_base"* %"i0'")
; CHECK-NEXT: invert:
; CHECK-NEXT:   %i1 = call %"struct.std::_Rb_tree_node_base"* @_ZSt18_Rb_tree_incrementPKSt18_Rb_tree_node_base(%"struct.std::_Rb_tree_node_base"* %i0)
; CHECK-NEXT:   %i2 = call %"struct.std::_Rb_tree_node_base"* @_ZSt18_Rb_tree_decrementPKSt18_Rb_tree_node_base(%"struct.std::_Rb_tree_node_base"* %i1)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
