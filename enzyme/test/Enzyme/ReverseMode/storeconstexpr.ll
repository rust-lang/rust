; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -simplifycfg -instcombine -adce -S | FileCheck %s

@.str = private unnamed_addr constant [18 x i8] c"W(o=%d, i=%d)=%f\0A\00", align 1

define void @derivative(i64* %from, i64* %fromp, i64* %to, i64* %top) {
entry:
  %call = call double (i8*, ...) @__enzyme_autodiff(i8* bitcast (void (i64*, i64*)* @callee to i8*), metadata !"enzyme_dup", i64* %from, i64* %fromp, metadata !"enzyme_dup", i64* %to, i64* %top)
  ret void
}

define void @callee(i64* %from, i64* %to) {
entry:
  store i64 ptrtoint ([18 x i8]* @.str to i64), i64* %to
  ret void
}

; Function Attrs: alwaysinline
declare double @__enzyme_autodiff(i8*, ...)

; CHECK: define internal void @diffecallee(i64* %from, i64* %"from'", i64* %to, i64* %"to'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   store i64 ptrtoint ([18 x i8]* @.str to i64), i64* %"to'", align 4
; CHECK-NEXT:   store i64 ptrtoint ([18 x i8]* @.str to i64), i64* %to, align 4
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
