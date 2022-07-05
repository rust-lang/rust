; RUN: if [ %llvmver -ge 12 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi

%struct.complex = type { double, double }
%struct.TapeAndComplex = type { i8*, %struct.complex }

@__enzyme_register_gradient1 = dso_local local_unnamed_addr global [3 x i8*] [i8* bitcast ({ double, double } (%struct.complex*, %struct.complex*, i32)* @myblas_cdot to i8*), i8* bitcast (void (%struct.TapeAndComplex*, %struct.complex*, %struct.complex*, %struct.complex*, %struct.complex*, i32, i32)* @myblas_cdot_fwd to i8*), i8* bitcast (void (%struct.complex*, %struct.complex*, %struct.complex*, %struct.complex*, i32, i32, {double, double}, i8*)* @myblas_cdot_rev to i8*)], align 16
@__enzyme_register_gradient2 = dso_local local_unnamed_addr global [3 x i8*] [i8* bitcast (double (double, double)* @myblas_cabs to i8*), i8* bitcast ({ i8*, double } (double, double)* @myblas_cabs_fwd to i8*), i8* bitcast ({ double, double } (double, double, double, i8*)* @myblas_cabs_rev to i8*)], align 16

declare dso_local { double, double } @myblas_cdot(%struct.complex*, %struct.complex*, i32)

declare dso_local void @myblas_cdot_fwd(%struct.TapeAndComplex* sret(%struct.TapeAndComplex) align 8, %struct.complex*, %struct.complex*, %struct.complex*, %struct.complex*, i32, i32)

declare dso_local void @myblas_cdot_rev(%struct.complex*, %struct.complex*, %struct.complex*, %struct.complex*, i32, i32, { double, double }, i8*) 

declare dso_local double @myblas_cabs(double, double) 

declare dso_local { i8*, double } @myblas_cabs_fwd(double, double) 

declare dso_local { double, double } @myblas_cabs_rev(double, double, double, i8*) 

; Function Attrs: nounwind uwtable
define dso_local double @dotabs(%struct.complex* %a0, %struct.complex* %a1, i32 %a2) {
  %a4 = tail call { double, double } @myblas_cdot(%struct.complex* %a0, %struct.complex* %a1, i32 %a2)
  %a5 = extractvalue { double, double } %a4, 0
  %a6 = extractvalue { double, double } %a4, 1
  %r = fadd double %a5, %a6
  %a7 = tail call fast double @myblas_cabs(double %a5, double %a6)
  ret double %a7
}

define void @f(i8* %a, i8* %b, i8* %c, i8* %d, i32 %e) {
  call void (...) @__enzyme_autodiff(i8* bitcast (double (%struct.complex*, %struct.complex*, i32)* @dotabs to i8*), i8* %a, i8* %b, i8* %c, i8* %d, i32 %e)
  ret void
}

declare void @__enzyme_autodiff(...)

; CHECK: define internal void @diffedotabs(%struct.complex* %a0, %struct.complex* %"a0'", %struct.complex* %a1, %struct.complex* %"a1'", i32 %a2, double %differeturn)
; CHECK-NEXT: invert:
; CHECK-NEXT:   %"a4'de" = alloca { double, double }, align 8
; CHECK-NEXT:   store { double, double } zeroinitializer, { double, double }* %"a4'de", align 8
; CHECK-NEXT:   %a4_augmented = call { i8*, { double, double } } @fixaugment_myblas_cdot(%struct.complex* %a0, %struct.complex* %"a0'", %struct.complex* %a1, %struct.complex* %"a1'", i32 %a2)
; CHECK-NEXT:   %subcache = extractvalue { i8*, { double, double } } %a4_augmented, 0
; CHECK-NEXT:   %a4 = extractvalue { i8*, { double, double } } %a4_augmented, 1
; CHECK-NEXT:   %a5 = extractvalue { double, double } %a4, 0
; CHECK-NEXT:   %a6 = extractvalue { double, double } %a4, 1
; CHECK-NEXT:   %0 = call { double, double } @fixgradient_myblas_cabs(double %a5, double %a6, double %differeturn)
; CHECK-NEXT:   %1 = extractvalue { double, double } %0, 0
; CHECK-NEXT:   %2 = extractvalue { double, double } %0, 1
; CHECK-NEXT:   %3 = getelementptr inbounds { double, double }, { double, double }* %"a4'de", i32 0, i32 1
; CHECK-NEXT:   %4 = load double, double* %3, align 8
; CHECK-NEXT:   %5 = fadd fast double %4, %2
; CHECK-NEXT:   store double %5, double* %3, align 8
; CHECK-NEXT:   %6 = getelementptr inbounds { double, double }, { double, double }* %"a4'de", i32 0, i32 0
; CHECK-NEXT:   %7 = load double, double* %6, align 8
; CHECK-NEXT:   %8 = fadd fast double %7, %1
; CHECK-NEXT:   store double %8, double* %6, align 8
; CHECK-NEXT:   %9 = load { double, double }, { double, double }* %"a4'de", align 8
; CHECK-NEXT:   call void @fixgradient_myblas_cdot(%struct.complex* %a0, %struct.complex* %"a0'", %struct.complex* %a1, %struct.complex* %"a1'", i32 %a2, { double, double } %9, i8* %subcache)
; CHECK-NEXT:   store { double, double } zeroinitializer, { double, double }* %"a4'de", align 8
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

