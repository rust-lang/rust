;RUN: %opt < %s %loadEnzyme -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare dso_local void @__enzyme_autodiff(...)

declare double @cblas_ddot(i32, double*, i32, double*, i32)

define void @active(i32 %len, double* noalias %m, double* %dm, i32 %incm, double* noalias %n, double* %dn, i32 %incn) {
entry:
  call void (...) @__enzyme_autodiff(double (i32, double*, i32, double*, i32)* @f, i32 %len, double* noalias %m, double* %dm, i32 %incm, double* noalias %n, double* %dn, i32 %incn)
  ret void
}

define void @inactiveFirst(i32 %len, double* noalias %m, i32 %incm, double* noalias %n, double* %dn, i32 %incn) {
entry:
  call void (...) @__enzyme_autodiff(double (i32, double*, i32, double*, i32)* @f, i32 %len, metadata !"enzyme_const", double* noalias %m, i32 %incm, double* noalias %n, double* %dn, i32 %incn)
  ret void
}

define void @inactiveSecond(i32 %len, double* noalias %m, double* noalias %dm, i32 %incm, double* noalias %n, i32 %incn) {
entry:
  call void (...) @__enzyme_autodiff(double (i32, double*, i32, double*, i32)* @f, i32 %len, double* noalias %m, double* noalias %dm, i32 %incm, metadata !"enzyme_const", double* noalias %n, i32 %incn)
  ret void
}

define void @activeMod(i32 %len, double* noalias %m, double* %dm, i32 %incm, double* noalias %n, double* %dn, i32 %incn) {
entry:
  call void (...) @__enzyme_autodiff(double (i32, double*, i32, double*, i32)* @modf, i32 %len, double* noalias %m, double* %dm, i32 %incm, double* noalias %n, double* %dn, i32 %incn)
  ret void
}

define void @inactiveModFirst(i32 %len, double* noalias %m, i32 %incm, double* noalias %n, double* %dn, i32 %incn) {
entry:
  call void (...) @__enzyme_autodiff(double (i32, double*, i32, double*, i32)* @modf, i32 %len, metadata !"enzyme_const", double* noalias %m, i32 %incm, double* noalias %n, double* %dn, i32 %incn)
  ret void
}

define void @inactiveModSecond(i32 %len, double* noalias %m, double* noalias %dm, i32 %incm, double* noalias %n, i32 %incn) {
entry:
  call void (...) @__enzyme_autodiff(double (i32, double*, i32, double*, i32)* @modf, i32 %len, double* noalias %m, double* noalias %dm, i32 %incm, metadata !"enzyme_const", double* noalias %n, i32 %incn)
  ret void
}

define double @f(i32 %len, double* noalias %m, i32 %incm, double* noalias %n, i32 %incn) {
entry:
  %call = call double @cblas_ddot(i32 %len, double* %m, i32 %incm, double* %n, i32 %incn)
  ret double %call
}

define double @modf(i32 %len, double* noalias %m, i32 %incm, double* noalias %n, i32 %incn) {
entry:
  %call = call double @f(i32 %len, double* %m, i32 %incm, double* %n, i32 %incn)
  store double 0.000000e+00, double* %m
  store double 0.000000e+00, double* %n
  ret double %call
}


; CHECK: define void @active
; CHECK-NEXT: entry
; CHECK-NEXT: call void @[[active:.+]](

; CHECK: define void @inactiveFirst
; CHECK-NEXT: entry
; CHECK-NEXT: call void @[[inactiveFirst:.+]](

; CHECK: define void @inactiveSecond
; CHECK-NEXT: entry
; CHECK-NEXT: call void @[[inactiveSecond:.+]](


; CHECK: define void @activeMod
; CHECK-NEXT: entry
; CHECK-NEXT: call void @[[activeMod:.+]](

; CHECK: define void @inactiveModFirst
; CHECK-NEXT: entry
; CHECK-NEXT: call void @[[inactiveModFirst:.+]](

; CHECK: define void @inactiveModSecond
; CHECK-NEXT: entry
; CHECK-NEXT: call void @[[inactiveModSecond:.+]](


; CHECK: define internal void @[[active]](i32 %len, double* noalias %m, double* %"m'", i32 %incm, double* noalias %n, double* %"n'", i32 %incn, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @cblas_daxpy(i32 %len, double %differeturn, double* %m, i32 %incm, double* %"n'", i32 %incn)
; CHECK-NEXT:   call void @cblas_daxpy(i32 %len, double %differeturn, double* %n, i32 %incn, double* %"m'", i32 %incm)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @[[inactiveFirst]](i32 %len, double* noalias %m, i32 %incm, double* noalias %n, double* %"n'", i32 %incn, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @cblas_daxpy(i32 %len, double %differeturn, double* %m, i32 %incm, double* %"n'", i32 %incn)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @[[inactiveSecond]](i32 %len, double* noalias %m, double* %"m'", i32 %incm, double* noalias %n, i32 %incn, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @cblas_daxpy(i32 %len, double %differeturn, double* %n, i32 %incn, double* %"m'", i32 %incm)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @[[activeMod]](i32 %len, double* noalias %m, double* %"m'", i32 %incm, double* noalias %n, double* %"n'", i32 %incn, double %differeturn)
; CHECK-NEXT: entry:
; CHECK:        %call_augmented = call { double*, double* } @[[augMod:.+]](i32 %len, double* %m, double* %"m'", i32 %incm, double* %n, double* %"n'", i32 %incn)
; CHECK:        call void @[[revMod:.+]](i32 %len, double* %m, double* %"m'", i32 %incm, double* %n, double* %"n'", i32 %incn, double %differeturn, { double*, double* } %call_augmented)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal { double*, double* } @[[augMod]](i32 %len, double* noalias %m, double* %"m'", i32 %incm, double* noalias %n, double* %"n'", i32 %incn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %mallocsize = mul i32 %len, 8
; CHECK-NEXT:   %malloccall = tail call i8* @malloc(i32 %mallocsize)
; CHECK-NEXT:   %0 = bitcast i8* %malloccall to double*
; CHECK-NEXT:   %1 = icmp eq i32 %len, 0
; CHECK-NEXT:   br i1 %1, label %__enzyme_memcpy_double_32_da0sa0stride.exit, label %for.body.i

; CHECK: for.body.i:                                       ; preds = %for.body.i, %entry
; CHECK-NEXT:   %idx.i = phi i32 [ 0, %entry ], [ %idx.next.i, %for.body.i ]
; CHECK-NEXT:   %sidx.i = phi i32 [ 0, %entry ], [ %sidx.next.i, %for.body.i ]
; CHECK-NEXT:   %dst.i.i = getelementptr inbounds double, double* %0, i32 %idx.i
; CHECK-NEXT:   %src.i.i = getelementptr inbounds double, double* %m, i32 %sidx.i
; CHECK-NEXT:   %src.i.l.i = load double, double* %src.i.i
; CHECK-NEXT:   store double %src.i.l.i, double* %dst.i.i
; CHECK-NEXT:   %idx.next.i = add nuw i32 %idx.i, 1
; CHECK-NEXT:   %sidx.next.i = add nuw i32 %sidx.i, %incm
; CHECK-NEXT:   %2 = icmp eq i32 %len, %idx.next.i
; CHECK-NEXT:   br i1 %2, label %__enzyme_memcpy_double_32_da0sa0stride.exit, label %for.body.i

; CHECK: __enzyme_memcpy_double_32_da0sa0stride.exit:      ; preds = %entry, %for.body.i
; CHECK-NEXT:   %mallocsize1 = mul i32 %len, 8
; CHECK-NEXT:   %malloccall2 = tail call i8* @malloc(i32 %mallocsize1)
; CHECK-NEXT:   %3 = bitcast i8* %malloccall2 to double*
; CHECK-NEXT:   %4 = icmp eq i32 %len, 0
; CHECK-NEXT:   br i1 %4, label %__enzyme_memcpy_double_32_da0sa0stride.exit9, label %for.body.i8

; CHECK: for.body.i8:                                      ; preds = %for.body.i8, %__enzyme_memcpy_double_32_da0sa0stride.exit
; CHECK-NEXT:   %idx.i1 = phi i32 [ 0, %__enzyme_memcpy_double_32_da0sa0stride.exit ], [ %idx.next.i6, %for.body.i8 ]
; CHECK-NEXT:   %sidx.i2 = phi i32 [ 0, %__enzyme_memcpy_double_32_da0sa0stride.exit ], [ %sidx.next.i7, %for.body.i8 ]
; CHECK-NEXT:   %dst.i.i3 = getelementptr inbounds double, double* %3, i32 %idx.i1
; CHECK-NEXT:   %src.i.i4 = getelementptr inbounds double, double* %n, i32 %sidx.i2
; CHECK-NEXT:   %src.i.l.i5 = load double, double* %src.i.i4
; CHECK-NEXT:   store double %src.i.l.i5, double* %dst.i.i3
; CHECK-NEXT:   %idx.next.i6 = add nuw i32 %idx.i1, 1
; CHECK-NEXT:   %sidx.next.i7 = add nuw i32 %sidx.i2, %incn
; CHECK-NEXT:   %5 = icmp eq i32 %len, %idx.next.i6
; CHECK-NEXT:   br i1 %5, label %__enzyme_memcpy_double_32_da0sa0stride.exit9, label %for.body.i8

; CHECK: __enzyme_memcpy_double_32_da0sa0stride.exit9:     ; preds = %__enzyme_memcpy_double_32_da0sa0stride.exit, %for.body.i8
; CHECK-NEXT:   %6 = insertvalue { double*, double* } undef, double* %0, 0
; CHECK-NEXT:   %7 = insertvalue { double*, double* } %6, double* %3, 1
; CHECK-NEXT:   ret { double*, double* } %7
; CHECK-NEXT: }

; CHECK: define internal void @[[revMod]](i32 %len, double* noalias %m, double* %"m'", i32 %incm, double* noalias %n, double* %"n'", i32 %incn, double %differeturn, { double*, double* }
; CHECK-NEXT: entry:
; CHECK-NEXT:   %1 = extractvalue { double*, double* } %0, 0
; CHECK-NEXT:   %2 = extractvalue { double*, double* } %0, 1
; CHECK-NEXT:   call void @cblas_daxpy(i32 %len, double %differeturn, double* %1, i32 1, double* %"n'", i32 %incn)
; CHECK-NEXT:   call void @cblas_daxpy(i32 %len, double %differeturn, double* %2, i32 1, double* %"m'", i32 %incm)
; CHECK-NEXT:   %3 = bitcast double* %1 to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %3)
; CHECK-NEXT:   %4 = bitcast double* %2 to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %4)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @[[inactiveModFirst]](i32 %len, double* noalias %m, i32 %incm, double* noalias %n, double* %"n'", i32 %incn, double %differeturn)
; CHECK-NEXT: entry:
; CHECK:        %call_augmented = call double* @[[augModFirst:.+]](i32 %len, double* %m, i32 %incm, double* %n, double* %"n'", i32 %incn)
; CHECK:        call void @[[revModFirst:.+]](i32 %len, double* %m, i32 %incm, double* %n, double* %"n'", i32 %incn, double %differeturn, double* %call_augmented)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal double* @augmented_f.6(i32 %len, double* noalias %m, i32 %incm, double* noalias %n, double* %"n'", i32 %incn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %mallocsize = mul i32 %len, 8
; CHECK-NEXT:   %malloccall = tail call i8* @malloc(i32 %mallocsize)
; CHECK-NEXT:   %0 = bitcast i8* %malloccall to double*
; CHECK-NEXT:   %1 = icmp eq i32 %len, 0
; CHECK-NEXT:   br i1 %1, label %__enzyme_memcpy_double_32_da0sa0stride.exit, label %for.body.i

; CHECK: for.body.i:                                       ; preds = %for.body.i, %entry
; CHECK-NEXT:   %idx.i = phi i32 [ 0, %entry ], [ %idx.next.i, %for.body.i ]
; CHECK-NEXT:   %sidx.i = phi i32 [ 0, %entry ], [ %sidx.next.i, %for.body.i ]
; CHECK-NEXT:   %dst.i.i = getelementptr inbounds double, double* %0, i32 %idx.i
; CHECK-NEXT:   %src.i.i = getelementptr inbounds double, double* %m, i32 %sidx.i
; CHECK-NEXT:   %src.i.l.i = load double, double* %src.i.i
; CHECK-NEXT:   store double %src.i.l.i, double* %dst.i.i
; CHECK-NEXT:   %idx.next.i = add nuw i32 %idx.i, 1
; CHECK-NEXT:   %sidx.next.i = add nuw i32 %sidx.i, %incm
; CHECK-NEXT:   %2 = icmp eq i32 %len, %idx.next.i
; CHECK-NEXT:   br i1 %2, label %__enzyme_memcpy_double_32_da0sa0stride.exit, label %for.body.i

; CHECK: __enzyme_memcpy_double_32_da0sa0stride.exit:      ; preds = %entry, %for.body.i
; CHECK-NEXT:   ret double* %0
; CHECK-NEXT: }

; CHECK: define internal void @[[revModFirst]](i32 %len, double* noalias %m, i32 %incm, double* noalias %n, double* %"n'", i32 %incn, double %differeturn, double*
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @cblas_daxpy(i32 %len, double %differeturn, double* %0, i32 1, double* %"n'", i32 %incn)
; CHECK-NEXT:   %1 = bitcast double* %0 to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %1)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @[[inactiveModSecond]](i32 %len, double* noalias %m, double* %"m'", i32 %incm, double* noalias %n, i32 %incn, double %differeturn)
; CHECK-NEXT: entry:
; CHECK:        %call_augmented = call double* @[[augModSecond:.+]](i32 %len, double* %m, double* %"m'", i32 %incm, double* %n, i32 %incn)
; CHECK:        call void @[[revModSecond:.+]](i32 %len, double* %m, double* %"m'", i32 %incm, double* %n, i32 %incn, double %differeturn, double* %call_augmented)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal double* @[[augModSecond]](i32 %len, double* noalias %m, double* %"m'", i32 %incm, double* noalias %n, i32 %incn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %mallocsize = mul i32 %len, 8
; CHECK-NEXT:   %malloccall = tail call i8* @malloc(i32 %mallocsize)
; CHECK-NEXT:   %0 = bitcast i8* %malloccall to double*
; CHECK-NEXT:   %1 = icmp eq i32 %len, 0
; CHECK-NEXT:   br i1 %1, label %__enzyme_memcpy_double_32_da0sa0stride.exit, label %for.body.i

; CHECK: for.body.i:                                       ; preds = %for.body.i, %entry
; CHECK-NEXT:   %idx.i = phi i32 [ 0, %entry ], [ %idx.next.i, %for.body.i ]
; CHECK-NEXT:   %sidx.i = phi i32 [ 0, %entry ], [ %sidx.next.i, %for.body.i ]
; CHECK-NEXT:   %dst.i.i = getelementptr inbounds double, double* %0, i32 %idx.i
; CHECK-NEXT:   %src.i.i = getelementptr inbounds double, double* %n, i32 %sidx.i
; CHECK-NEXT:   %src.i.l.i = load double, double* %src.i.i
; CHECK-NEXT:   store double %src.i.l.i, double* %dst.i.i
; CHECK-NEXT:   %idx.next.i = add nuw i32 %idx.i, 1
; CHECK-NEXT:   %sidx.next.i = add nuw i32 %sidx.i, %incn
; CHECK-NEXT:   %2 = icmp eq i32 %len, %idx.next.i
; CHECK-NEXT:   br i1 %2, label %__enzyme_memcpy_double_32_da0sa0stride.exit, label %for.body.i

; CHECK: __enzyme_memcpy_double_32_da0sa0stride.exit:      ; preds = %entry, %for.body.i
; CHECK-NEXT:   ret double* %0
; CHECK-NEXT: }

; CHECK: define internal void @[[revModSecond]](i32 %len, double* noalias %m, double* %"m'", i32 %incm, double* noalias %n, i32 %incn, double %differeturn, double*
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @cblas_daxpy(i32 %len, double %differeturn, double* %0, i32 1, double* %"m'", i32 %incm)
; CHECK-NEXT:   %1 = bitcast double* %0 to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %1)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

