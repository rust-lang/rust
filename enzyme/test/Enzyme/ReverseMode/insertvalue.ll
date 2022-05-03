; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

; Function Attrs: noinline nounwind readnone uwtable
define double @tester(double %x) {
entry:
  %agg1 = insertvalue [3 x double] undef, double %x, 0
  %mul = fmul double %x, %x
  %agg2 = insertvalue [3 x double] %agg1, double %mul, 1
  %add = fadd double %mul, 2.0
  %agg3 = insertvalue [3 x double] %agg2, double %add, 2
  %res = extractvalue [3 x double] %agg2, 1
  ret double %res
}

define double @test_derivative(double %x) {
entry:
  %0 = tail call double (double (double)*, ...) @__enzyme_autodiff(double (double)* nonnull @tester, double %x)
  ret double %0
}

; Function Attrs: nounwind
declare double @__enzyme_autodiff(double (double)*, ...)

; CHECK: define internal { double } @diffetester(double %x, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"agg2'de" = alloca [3 x double], align 8
; CHECK-NEXT:   store [3 x double] zeroinitializer, [3 x double]* %"agg2'de"
; CHECK-NEXT:   %"agg1'de" = alloca [3 x double], align 8
; CHECK-NEXT:   store [3 x double] zeroinitializer, [3 x double]* %"agg1'de"
; CHECK-NEXT:   %0 = getelementptr inbounds [3 x double], [3 x double]* %"agg2'de", i32 0, i32 1
; CHECK-NEXT:   %1 = load double, double* %0
; CHECK-NEXT:   %2 = fadd fast double %1, %differeturn
; CHECK-NEXT:   store double %2, double* %0
; CHECK-NEXT:   %3 = load [3 x double], [3 x double]* %"agg2'de"
; CHECK-NEXT:   %4 = extractvalue [3 x double] %3, 1
; CHECK-NEXT:   %[[i5:.+]] = load [3 x double], [3 x double]* %"agg2'de"
; CHECK-NEXT:   %[[i7:.+]] = extractvalue [3 x double] %[[i5]], 0
; CHECK-NEXT:   %[[i8:.+]] = getelementptr inbounds [3 x double], [3 x double]* %"agg1'de", i32 0, i32 0
; CHECK-NEXT:   %[[i9:.+]] = load double, double* %[[i8]]
; CHECK-NEXT:   %[[i10:.+]] = fadd fast double %[[i9]], %[[i7]]
; CHECK-NEXT:   store double %[[i10]], double* %[[i8]]
; CHECK-NEXT:   %[[i11:.+]] = getelementptr inbounds [3 x double], [3 x double]* %"agg1'de", i32 0, i32 1
; CHECK-NEXT:   %[[i12:.+]] = load double, double* %[[i11]]
; CHECK-NEXT:   store double %[[i12]], double* %[[i11]]
; CHECK-NEXT:   %[[i13:.+]] = extractvalue [3 x double] %[[i5]], 2
; CHECK-NEXT:   %[[i14:.+]] = getelementptr inbounds [3 x double], [3 x double]* %"agg1'de", i32 0, i32 2
; CHECK-NEXT:   %[[i15:.+]] = load double, double* %[[i14]]
; CHECK-NEXT:   %[[i16:.+]] = fadd fast double %[[i15]], %[[i13]]
; CHECK-NEXT:   store double %[[i16]], double* %[[i14]]
; CHECK-NEXT:   store [3 x double] zeroinitializer, [3 x double]* %"agg2'de"
; CHECK-NEXT:   %m0diffex = fmul fast double %4, %x
; CHECK-NEXT:   %m1diffex = fmul fast double %4, %x
; CHECK-NEXT:   %[[i17:.+]] = fadd fast double %m0diffex, %m1diffex
; CHECK-NEXT:   %[[i18:.+]] = load [3 x double], [3 x double]* %"agg1'de"
; CHECK-NEXT:   %[[i19:.+]] = extractvalue [3 x double] %[[i18]], 0
; CHECK-NEXT:   %[[i20:.+]] = fadd fast double %[[i17]], %[[i19]]
; CHECK-NEXT:   store [3 x double] zeroinitializer, [3 x double]* %"agg1'de"
; CHECK-NEXT:   %[[i21:.+]] = insertvalue { double } undef, double %[[i20]], 0
; CHECK-NEXT:   ret { double } %[[i21]]
; CHECK-NEXT: }
