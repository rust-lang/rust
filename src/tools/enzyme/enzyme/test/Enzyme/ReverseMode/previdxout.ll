; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -simplifycfg -adce -S | FileCheck %s
; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | %lli - | FileCheck %s --check-prefix=EVAL

; EVAL: 1.00

@a0 = private unnamed_addr constant [8 x i8] c"res=%f\0A\00", align 1

declare i8* @malloc(i64)

declare double @__enzyme_autodiff(i8*, double, i64)

declare void @printf(i8*, double)

define i64 @main() {
bb:
  %i = tail call double @__enzyme_autodiff(i8* bitcast (double (double, i64)* @f to i8*), double 2.000000e+00, i64 2)
  call void @printf(i8* nonnull dereferenceable(1) getelementptr inbounds ([8 x i8], [8 x i8]* @a0, i64 0, i64 0), double %i)
  ret i64 0
}

define double @f(double %arg, i64 %arg1) {
bb:
  %i2 = alloca i8, i64 8, align 1
  %i6 = bitcast i8* %i2 to double*
  br i1 true, label %pre, label %bb18

pre:                                    ; preds = %bb
  br label %bb9

bb9:                                              ; preds = %bb14, %bb7
  %i10 = phi i64 [ %i13, %bb14 ], [ 0, %pre ]
  %i11 = phi i1 [ true, %bb14 ], [ false, %pre ]
  %i13 = add nuw nsw i64 %i10, 1
  br i1 %i11, label %bb17, label %bb14

bb14:                                             ; preds = %bb9
  store double %arg, double* %i6, align 1
  br i1 true, label %bb17, label %bb9

bb17:                                             ; preds = %bb14, %bb9
  br label %bb18

bb18:                                             ; preds = %bb17, %bb5
  %i20 = load double, double* %i6, align 1
  ret double %i20
}

; CHECK: define internal { double } @diffef(double %arg, i64 %arg1, double %differeturn)
; CHECK-NEXT: bb:
; CHECK-NEXT:   %"i2'ipa" = alloca i8, i64 8, align 1
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* nonnull align 1 %"i2'ipa", i8 0, i64 8, i1 false)
; CHECK-NEXT:   %i2 = alloca i8, i64 8, align 1
; CHECK-NEXT:   %"i6'ipc" = bitcast i8* %"i2'ipa" to double*
; CHECK-NEXT:   %i6 = bitcast i8* %i2 to double*
; CHECK-NEXT:   store double %arg, double* %i6, align 1
; CHECK-NEXT:   %0 = load double, double* %"i6'ipc", align 1
; CHECK-NEXT:   %1 = fadd fast double %0, %differeturn
; CHECK-NEXT:   store double %1, double* %"i6'ipc", align 1
; CHECK-NEXT:   %2 = icmp ne i64 0, 0
; CHECK-NEXT:   %i11_unwrap = select i1 %2, i1 true, i1 false
; CHECK-NEXT:   br i1 %i11_unwrap, label %invertbb9, label %invertbb14

; CHECK: invertpre:                                        ; preds = %invertbb9
; CHECK-NEXT:   %3 = insertvalue { double } undef, double %"arg'de.1", 0
; CHECK-NEXT:   ret { double } %3

; CHECK: invertbb9:                                        ; preds = %bb, %invertbb14
; CHECK-NEXT:   %"arg'de.1" = phi double [ %7, %invertbb14 ], [ 0.000000e+00, %bb ]
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ %"iv'ac.1", %invertbb14 ], [ 0, %bb ]
; CHECK-NEXT:   %4 = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   br i1 %4, label %invertpre, label %incinvertbb9

; CHECK: incinvertbb9:                                     ; preds = %invertbb9
; CHECK-NEXT:   %5 = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %invertbb14

; CHECK: invertbb14:                                       ; preds = %bb, %incinvertbb9
; CHECK-NEXT:   %"arg'de.2" = phi double [ %"arg'de.1", %incinvertbb9 ], [ 0.000000e+00, %bb ]
; CHECK-NEXT:   %"iv'ac.1" = phi i64 [ %5, %incinvertbb9 ], [ 0, %bb ]
; CHECK-NEXT:   %6 = load double, double* %"i6'ipc", align 1
; CHECK-NEXT:   store double 0.000000e+00, double* %"i6'ipc", align 1
; CHECK-NEXT:   %7 = fadd fast double %"arg'de.2", %6
; CHECK-NEXT:   br label %invertbb9
; CHECK-NEXT: }
