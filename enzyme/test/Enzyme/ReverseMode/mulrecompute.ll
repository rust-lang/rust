; RUN: if [ %llvmver -le 12 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S -gvn -dse -dse | FileCheck %s ; fi
; RUN: if [ %llvmver -ge 13 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S -gvn -dse -dse | FileCheck %s --check-prefix=POST ; fi

declare double @__enzyme_autodiff(...)

define double @outer(double %x, double %y) {
entry:
  %res = call double (...) @__enzyme_autodiff(i8* bitcast (double (double, double)* @julia_sphericalbesselj_672 to i8*), double %x, metadata !"enzyme_const", double %y)
  ret double %res
}

define double @julia_besselj_685(double %x, double %y) {
entry:
  ret double %x
}

define double @julia_sphericalbesselj_672(double %x, double %y) {
top:
  %c = call fastcc double @julia_besselj_685(double %x, double %y)
  %cmp = fcmp uge double %y, 0.000000e+00
  br i1 %cmp, label %L24, label %L22

L22:                                              ; preds = %L17
  call void @llvm.trap()
  unreachable

L24:                                              ; preds = %L17
  %sq = call double @llvm.sqrt.f64(double %y)
  %res = fmul double %sq, %c
  ret double %res
}

declare double @llvm.fabs.f64(double)
declare double @llvm.sqrt.f64(double)
declare void @llvm.trap()

; CHECK: define internal { double } @diffejulia_sphericalbesselj_672(double %x, double %y, double %differeturn)
; CHECK-NEXT: top:
; CHECK-NEXT:   %cmp = fcmp uge double %y, 0.000000e+00
; CHECK-NEXT:   br i1 %cmp, label %L24, label %L22

; CHECK: L22:                                              ; preds = %top
; CHECK-NEXT:   call void @llvm.trap()
; CHECK-NEXT:   unreachable

; CHECK: L24:                                              ; preds = %top
; CHECK-NEXT:   %sq = call double @llvm.sqrt.f64(double %y)
; CHECK-NEXT:   %m1diffec = fmul fast double %differeturn, %sq
; CHECK-NEXT:   %0 = call fastcc { double } @diffejulia_besselj_685(double %x, double %y, double %m1diffec)
; CHECK-NEXT:   ret { double } %0
; CHECK-NEXT: }

; POST: define internal { double } @diffejulia_sphericalbesselj_672(double %x, double %y, double %differeturn)
; POST-NEXT: top:
; POST-NEXT:   %cmp = fcmp uge double %y, 0.000000e+00
; POST-NEXT:   call void @llvm.assume(i1 %cmp)
; POST-NEXT:   %sq = call double @llvm.sqrt.f64(double %y)
; POST-NEXT:   %m1diffec = fmul fast double %differeturn, %sq
; POST-NEXT:   %0 = call fastcc { double } @diffejulia_besselj_685(double %x, double %y, double %m1diffec)
; POST-NEXT:   ret { double } %0
; POST-NEXT: }
