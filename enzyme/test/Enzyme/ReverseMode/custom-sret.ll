; RUN: if [ %llvmver -ge 12 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi

%struct.complex = type { double, double }
%struct.TapeAndComplex = type { i8*, %struct.complex }

@__enzyme_register_gradient1 = global [3 x i8*] [i8* bitcast ({ double, double } (%struct.complex*, %struct.complex*, i32)* @myblas_cdot to i8*), i8* bitcast (void (%struct.TapeAndComplex*, %struct.complex*, %struct.complex*, %struct.complex*, %struct.complex*, i32, i32)* @myblas_cdot_fwd to i8*), i8* bitcast (void (%struct.complex*, %struct.complex*, %struct.complex*, %struct.complex*, i32, i32, { double, double }, i8*)* @myblas_cdot_rev to i8*)], align 16

declare { double, double } @myblas_cdot(%struct.complex*, %struct.complex*, i32)

declare void @myblas_cdot_fwd(%struct.TapeAndComplex* sret(%struct.TapeAndComplex) align 8, %struct.complex*, %struct.complex*, %struct.complex*, %struct.complex*, i32, i32)

declare void @myblas_cdot_rev(%struct.complex*, %struct.complex*, %struct.complex*, %struct.complex*, i32, i32, { double, double }, i8*)

define double @dotabs(%struct.complex* %a0, %struct.complex* %a1, i32 %a2) {
  %a4 = call { double, double } @myblas_cdot(%struct.complex* %a0, %struct.complex* %a1, i32 %a2)
  %a5 = extractvalue { double, double } %a4, 0
  %a6 = extractvalue { double, double } %a4, 1
  %r = fadd double %a5, %a6
  ret double %r
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
; CHECK-NEXT:   %0 = getelementptr inbounds { double, double }, { double, double }* %"a4'de", i32 0, i32 1
; CHECK-NEXT:   %1 = load double, double* %0, align 8
; CHECK-NEXT:   %2 = fadd fast double %1, %differeturn
; CHECK-NEXT:   store double %2, double* %0, align 8
; CHECK-NEXT:   %3 = getelementptr inbounds { double, double }, { double, double }* %"a4'de", i32 0, i32 0
; CHECK-NEXT:   %4 = load double, double* %3, align 8
; CHECK-NEXT:   %5 = fadd fast double %4, %differeturn
; CHECK-NEXT:   store double %5, double* %3, align 8
; CHECK-NEXT:   %6 = load { double, double }, { double, double }* %"a4'de", align 8
; CHECK-NEXT:   call void @fixgradient_myblas_cdot(%struct.complex* %a0, %struct.complex* %"a0'", %struct.complex* %a1, %struct.complex* %"a1'", i32 %a2, { double, double } %6)
; CHECK-NEXT:   store { double, double } zeroinitializer, { double, double }* %"a4'de", align 8
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @fixgradient_myblas_cdot(%struct.complex* %arg0, %struct.complex* %arg1, %struct.complex* %arg2, %struct.complex* %arg3, i32 %arg4, { double, double } %arg5) 
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call { i8*, { double, double } } @fixaugment_myblas_cdot(%struct.complex* %arg0, %struct.complex* %arg1, %struct.complex* %arg2, %struct.complex* %arg3, i32 %arg4)
; CHECK-NEXT:   %1 = extractvalue { i8*, { double, double } } %0, 0
; CHECK-NEXT:   call void @fixgradient_myblas_cdot.1(%struct.complex* %arg0, %struct.complex* %arg1, %struct.complex* %arg2, %struct.complex* %arg3, i32 %arg4, { double, double } %arg5, i8* %1)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal %struct.TapeAndComplex @fixaugment_myblas_cdot_fwd(%struct.complex* %arg0, %struct.complex* %arg1, %struct.complex* %arg2, %struct.complex* %arg3, i32 %arg4, i32 %arg5)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = alloca %struct.TapeAndComplex, align 8
; CHECK-NEXT:   call void @myblas_cdot_fwd(%struct.TapeAndComplex* %0, %struct.complex* %arg0, %struct.complex* %arg1, %struct.complex* %arg2, %struct.complex* %arg3, i32 %arg4, i32 %arg5)
; CHECK-NEXT:   %1 = load %struct.TapeAndComplex, %struct.TapeAndComplex* %0, align 8
; CHECK-NEXT:   ret %struct.TapeAndComplex %1
; CHECK-NEXT: }

; CHECK: define internal { i8*, { double, double } } @fixaugment_fixaugment_myblas_cdot_fwd(%struct.complex* %arg0, %struct.complex* %arg1, %struct.complex* %arg2, %struct.complex* %arg3, i32 %arg4, i32 %arg5) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call %struct.TapeAndComplex @fixaugment_myblas_cdot_fwd(%struct.complex* %arg0, %struct.complex* %arg1, %struct.complex* %arg2, %struct.complex* %arg3, i32 %arg4, i32 %arg5)
; CHECK-NEXT:   %1 = extractvalue %struct.TapeAndComplex %0, 0
; CHECK-NEXT:   %2 = insertvalue { i8*, { double, double } } undef, i8* %1, 0
; CHECK-NEXT:   %3 = alloca { double, double }
; CHECK-DAG:   %[[i4:.+]] = bitcast { double, double }* %3 to %struct.complex*
; CHECK-DAG:   %[[i5:.+]] = extractvalue %struct.TapeAndComplex %0, 1
; CHECK-NEXT:   store %struct.complex %[[i5]], %struct.complex* %[[i4]]
; CHECK-NEXT:   %6 = load { double, double }, { double, double }* %3
; CHECK-NEXT:   %7 = insertvalue { i8*, { double, double } } %2, { double, double } %6, 1
; CHECK-NEXT:   ret { i8*, { double, double } } %7
; CHECK-NEXT: }

; CHECK: define internal { i8*, { double, double } } @fixaugment_myblas_cdot(%struct.complex* %arg0, %struct.complex* %"arg0'", %struct.complex* %arg1, %struct.complex* %"arg1'", i32 %arg2)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call { i8*, { double, double } } @fixaugment_fixaugment_myblas_cdot_fwd(%struct.complex* %arg0, %struct.complex* %"arg0'", %struct.complex* %arg1, %struct.complex* %"arg1'", i32 %arg2, i32 %arg2)
; CHECK-NEXT:   ret { i8*, { double, double } } %0
; CHECK-NEXT: }

; CHECK: define internal void @fixgradient_myblas_cdot.1(%struct.complex* %arg0, %struct.complex* %"arg0'", %struct.complex* %arg1, %struct.complex* %"arg1'", i32 %arg2, { double, double } %postarg0, i8* %postarg1)
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @myblas_cdot_rev(%struct.complex* %arg0, %struct.complex* %"arg0'", %struct.complex* %arg1, %struct.complex* %"arg1'", i32 %arg2, i32 %arg2, { double, double } %postarg0, i8* %postarg1)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
