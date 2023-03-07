; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -simplifycfg -instsimplify -adce -S | FileCheck %s

@.str = private unnamed_addr constant [28 x i8] c"original =%f derivative=%f\0A\00", align 1

define dso_local double* @f(double** nocapture readonly %a0) local_unnamed_addr #0 {
entry:
  %a2 = load double*, double** %a0, align 8
  ret double* %a2
}

define dso_local double @submalloced(double* %a0) local_unnamed_addr #1 {
entry: 
  %p2 = call noalias i8* @malloc(i32 8)
  %p3 = bitcast i8* %p2 to double**
  store double* %a0, double** %p3, align 8
  %a4 = call double* @f(double** nonnull %p3)
  %r = load double, double* %a4
  call void @free(i8* %p2)
  ret double %r
}

declare dso_local noalias i8* @malloc(i32) local_unnamed_addr #2

declare dso_local void @free(i8* nocapture) local_unnamed_addr #3

define dso_local double @malloced(double* %a0) #1 {
entry:
  %a3 = call double @submalloced(double* %a0)
  %a4 = fmul double %a3, %a3
  ret double %a4
}

define dso_local void @derivative(double* %a0, double* %a1) local_unnamed_addr #4 {
entry:
  call void (i8*, ...) @__enzyme_autodiff(i8* bitcast (double (double*)* @malloced to i8*), double* %a0, double* %a1)
  ret void
}

declare dso_local void @__enzyme_autodiff(i8*, ...)

attributes #0 = { noinline norecurse nounwind readonly }
attributes #1 = { nounwind }
attributes #2 = { inaccessiblememonly nounwind }
attributes #3 = { inaccessiblemem_or_argmemonly nounwind }
attributes #4 = { nounwind }
attributes #6 = { nounwind readonly }
attributes #7 = { nounwind }
attributes #9 = { nounwind }

; CHECK: define internal { double*, double } @augmented_submalloced(double* %a0, double* %"a0'") 
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = alloca { double*, double }
; CHECK-NEXT:   %1 = getelementptr inbounds { double*, double }, { double*, double }* %0, i32 0, i32 0
; CHECK-NEXT:   %p2 = call noalias nonnull dereferenceable(8) dereferenceable_or_null(8) i8* @malloc(i32 8)
; CHECK-NEXT:   %"p2'mi" = call noalias nonnull dereferenceable(8) dereferenceable_or_null(8) i8* @malloc(i32 8)
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* nonnull dereferenceable(8) dereferenceable_or_null(8) %"p2'mi", i8 0, i64 8, i1 false)
; CHECK-NEXT:   %"p3'ipc" = bitcast i8* %"p2'mi" to double**
; CHECK-NEXT:   %p3 = bitcast i8* %p2 to double**
; CHECK-NEXT:   store double* %"a0'", double** %"p3'ipc", align 8, !alias.scope !0, !noalias !3
; CHECK-NEXT:   store double* %a0, double** %p3, align 8, !alias.scope !3, !noalias !0
; CHECK-NEXT:   %a4_augmented = call { double*, double* } @augmented_f(double** %p3, double** %"p3'ipc")
; CHECK-NEXT:   %a4 = extractvalue { double*, double* } %a4_augmented, 0
; CHECK-NEXT:   %"a4'ac" = extractvalue { double*, double* } %a4_augmented, 1
; CHECK-NEXT:   store double* %"a4'ac", double** %1
; CHECK-NEXT:   %r = load double, double* %a4
; CHECK-NEXT:   call void @free(i8* %p2)
; CHECK-NEXT:   call void @free(i8* %"p2'mi")
; CHECK-NEXT:   %2 = getelementptr inbounds { double*, double }, { double*, double }* %0, i32 0, i32 1
; CHECK-NEXT:   store double %r, double* %2
; CHECK-NEXT:   %3 = load { double*, double }, { double*, double }* %0
; CHECK-NEXT:   ret { double*, double } %3
; CHECK-NEXT: }

; CHECK: define internal void @diffesubmalloced(double* %a0, double* %"a0'", double %differeturn, double* %"a4'ip_phi") 
; CHECK-NEXT: entry:
; CHECK-NEXT:   %p2 = call noalias nonnull dereferenceable(8) dereferenceable_or_null(8) i8* @malloc(i32 8)
; CHECK-NEXT:   %"p2'mi" = call noalias nonnull dereferenceable(8) dereferenceable_or_null(8) i8* @malloc(i32 8)
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* nonnull dereferenceable(8) dereferenceable_or_null(8) %"p2'mi", i8 0, i64 8, i1 false)
; CHECK-NEXT:   %"p3'ipc" = bitcast i8* %"p2'mi" to double**
; CHECK-NEXT:   %p3 = bitcast i8* %p2 to double**
; CHECK-NEXT:   store double* %"a0'", double** %"p3'ipc", align 8
; CHECK-NEXT:   store double* %a0, double** %p3, align 8
; CHECK-NEXT:   %0 = load double, double* %"a4'ip_phi"
; CHECK-NEXT:   %1 = fadd fast double %0, %differeturn
; CHECK-NEXT:   store double %1, double* %"a4'ip_phi"
; CHECK-NEXT:   call void @diffef(double** %p3, double** %"p3'ipc")
; CHECK-NEXT:   tail call void @free(i8* nonnull %"p2'mi")
; CHECK-NEXT:   tail call void @free(i8* nonnull %p2)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
