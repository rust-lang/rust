; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -simplifycfg -S | FileCheck %s

declare i8* @_Znwm(i64 %s)

define void @ifunc(i8* %arg) {
bb:
  %i66 = call i8* @recur(i8* %arg)
  ret void
}

define i8* @recur(i8* %arg) {
bb: 
  %i = call i8* @_Znwm(i64 8)
  %i20 = call i8* @recur(i8* %i)
  %i24 = bitcast i8* %arg to i8**
  store i8* %i20, i8** %i24, align 8
  ret i8* %i
}

declare void @_Z17__enzyme_autodiff(...) 

define void @caller(i8* %a, i8* %b) {
  tail call void (...) @_Z17__enzyme_autodiff(void (i8*)* @ifunc, metadata !"enzyme_dup", i8* %a, i8* %b)
  ret void
}

; Ensure this compiles without segfaulting
; CHECK: define internal void @differecur(i8* %arg, i8* %"arg'")
