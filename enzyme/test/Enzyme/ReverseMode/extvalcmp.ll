; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

; ModuleID = 'wa.cpp'
source_filename = "wa.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@enzyme_dup = dso_local local_unnamed_addr global i32 0, align 4

; Function Attrs: norecurse nounwind readonly uwtable willreturn mustprogress
define double @subfn({ i1, double* } %agg) {
entry:
  %cmp = extractvalue { i1, double* } %agg, 0
  br i1 %cmp, label %lblock, label %end

lblock:                                         ; preds = %entry, %_Z13cubicSpline3ddd.exit
  %ptr = extractvalue { i1, double* } %agg, 1
  %ld = load double, double* %ptr, align 8
  %sq = fmul double %ld, %ld
  br label %end

end:                                        ; preds = %for.body
  %res = phi double [ %sq, %lblock ], [ 0.000000e+00, %entry ]
  ret double %res
}

define double @identity(double %x) #0 {
entry:
  ret double %x
}

define double @_Z3fooPdy({ i1, double* } %agg) {
    %zed = call double @subfn({ i1, double* } %agg)
    %res = call double @identity(double %zed)
    ret double %res
}

; Function Attrs: uwtable mustprogress
define void @_Z6callerPdS_y({ i1, double* } %agg, { i1, double* } %dagg) {
entry:
  %0 = load i32, i32* @enzyme_dup, align 4
  %call = tail call double @_Z17__enzyme_autodiffPviPdS0_iy(i8* bitcast (double ({i1, double*})* @_Z3fooPdy to i8*), i32 %0, { i1, double* } %agg, { i1, double* } %dagg)
  ret void
}

declare double @_Z17__enzyme_autodiffPviPdS0_iy(i8*, i32, { i1, double* }, { i1, double* })

attributes #0 = { readnone }

; CHECK: define internal void @diffesubfn({ i1, double* } %agg, { i1, double* } %"agg'", double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %cmp = extractvalue { i1, double* } %agg, 0
; CHECK-NEXT:   %0 = select {{(fast )?}}i1 %cmp, double %differeturn, double 0.000000e+00
; CHECK-NEXT:   br i1 %cmp, label %invertlblock, label %invertentry

; CHECK: invertentry:                                      ; preds = %entry, %invertlblock
; CHECK-NEXT:   ret void

; CHECK: invertlblock:                                     ; preds = %entry
; CHECK-NEXT:   %ptr_unwrap = extractvalue { i1, double* } %agg, 1
; CHECK-NEXT:   %ld_unwrap = load double, double* %ptr_unwrap, align 8, !invariant.group
; CHECK-NEXT:   %m0diffeld = fmul fast double %0, %ld_unwrap
; CHECK-NEXT:   %m1diffeld = fmul fast double %0, %ld_unwrap
; CHECK-NEXT:   %1 = fadd fast double %m0diffeld, %m1diffeld
; CHECK-NEXT:   %"ptr'ipev_unwrap" = extractvalue { i1, double* } %"agg'", 1
; CHECK-NEXT:   %2 = load double, double* %"ptr'ipev_unwrap"
; CHECK-NEXT:   %3 = fadd fast double %2, %1
; CHECK-NEXT:   store double %3, double* %"ptr'ipev_unwrap"
; CHECK-NEXT:   br label %invertentry
; CHECK-NEXT: }