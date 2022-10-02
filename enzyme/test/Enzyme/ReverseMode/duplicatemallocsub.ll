; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -simplifycfg -instsimplify -adce -S | FileCheck %s

@.str = private unnamed_addr constant [28 x i8] c"original =%f derivative=%f\0A\00", align 1

define dso_local double @f(double* nocapture readonly %a0) local_unnamed_addr #0 {
entry:
  %a2 = load double, double* %a0, align 8
  %m2 = fmul double %a2, %a2
  ret double %m2
}

define dso_local double @submalloced(double %a0) local_unnamed_addr #1 {
entry:
  %a2 = call noalias dereferenceable_or_null(8) i8* @malloc(i32 8)
  %a3 = bitcast i8* %a2 to double*
  store double %a0, double* %a3, align 8
  %a4 = call double @f(double* nonnull %a3)
  call void @free(i8* %a2)
  ret double %a4
}

declare dso_local noalias i8* @malloc(i32) local_unnamed_addr #2

declare dso_local void @free(i8* nocapture) local_unnamed_addr #3

define dso_local double @malloced(double %a0, i32 %a1) #1 {
entry:
  %a3 = call double @submalloced(double %a0)
  %a4 = fmul double %a3, %a3
  ret double %a4
}

define dso_local double @derivative(double %a0, i32 %a1) local_unnamed_addr #4 {
entry:
  %a3 = call double (i8*, ...) @__enzyme_autodiff(i8* bitcast (double (double, i32)* @malloced to i8*), double %a0, i32 %a1) #9
  ret double %a3
}

declare dso_local double @__enzyme_autodiff(i8*, ...)

attributes #0 = { noinline norecurse nounwind readonly }
attributes #1 = { nounwind }
attributes #2 = { inaccessiblememonly nounwind }
attributes #3 = { inaccessiblemem_or_argmemonly nounwind }
attributes #4 = { nounwind }
attributes #6 = { nounwind readonly }
attributes #7 = { nounwind }
attributes #9 = { nounwind }

; CHECK: define internal { double } @diffemalloced(double %a0, i32 %a1, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %a3 = call fast double @augmented_submalloced(double %a0)
; CHECK-NEXT:   %m0diffea3 = fmul fast double %differeturn, %a3
; CHECK-NEXT:   %m1diffea3 = fmul fast double %differeturn, %a3
; CHECK-NEXT:   %0 = fadd fast double %m0diffea3, %m1diffea3
; CHECK-NEXT:   %1 = call { double } @diffesubmalloced(double %a0, double %0)
; CHECK-NEXT:   ret { double } %1
; CHECK-NEXT: }

; CHECK: define internal double @augmented_f(double* nocapture readonly %a0, double* nocapture %"a0'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %a2 = load double, double* %a0, align 8
; CHECK-NEXT:   %m2 = fmul double %a2, %a2
; CHECK-NEXT:   ret double %m2
; CHECK-NEXT: }

; CHECK: define internal double @augmented_submalloced(double %a0)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %a2 = call noalias dereferenceable_or_null(8) i8* @malloc(i32 8)
; CHECK-NEXT:   %a3 = bitcast i8* %a2 to double*
; CHECK-NEXT:   store double %a0, double* %a3, align 8
; CHECK-NEXT:   %a4 = call fast double @augmented_f(double* %a3, double* undef)
; CHECK-NEXT:   call void @free(i8* %a2)
; CHECK-NEXT:   ret double %a4
; CHECK-NEXT: }

; CHECK: define internal { double } @diffesubmalloced(double %a0, double %differeturn) 
; CHECK-NEXT: entry:
; CHECK-NEXT:   %a2 = call noalias nonnull dereferenceable(8) dereferenceable_or_null(8) i8* @malloc(i32 8)
; CHECK-NEXT:   %"a2'mi" = call noalias nonnull dereferenceable(8) dereferenceable_or_null(8) i8* @malloc(i32 8)
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* nonnull dereferenceable(8) dereferenceable_or_null(8) %"a2'mi", i8 0, i64 8, i1 false)
; CHECK-NEXT:   %"a3'ipc" = bitcast i8* %"a2'mi" to double*
; CHECK-NEXT:   %a3 = bitcast i8* %a2 to double*
; CHECK-NEXT:   store double %a0, double* %a3, align 8
; CHECK-NEXT:   call void @diffef(double* %a3, double* %"a3'ipc", double %differeturn)
; CHECK-NEXT:   %0 = load double, double* %"a3'ipc", align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"a3'ipc"
; CHECK-NEXT:   tail call void @free(i8* nonnull %"a2'mi")
; CHECK-NEXT:   tail call void @free(i8* nonnull %a2)
; CHECK-NEXT:   %1 = insertvalue { double } undef, double %0, 0
; CHECK-NEXT:   ret { double } %1
; CHECK-NEXT: }

; CHECK: define internal void @diffef(double* nocapture readonly %a0, double* nocapture %"a0'", double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %a2 = load double, double* %a0, align 8
; CHECK-NEXT:   %m0diffea2 = fmul fast double %differeturn, %a2
; CHECK-NEXT:   %m1diffea2 = fmul fast double %differeturn, %a2
; CHECK-NEXT:   %0 = fadd fast double %m0diffea2, %m1diffea2
; CHECK-NEXT:   %1 = load double, double* %"a0'", align 8
; CHECK-NEXT:   %2 = fadd fast double %1, %0
; CHECK-NEXT:   store double %2, double* %"a0'", align 8
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
