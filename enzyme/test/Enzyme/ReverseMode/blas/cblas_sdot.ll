;RUN: %opt < %s %loadEnzyme -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare dso_local void @__enzyme_autodiff(...)

declare float @cblas_sdot(i32, float*, i32, float*, i32)

define void @active(i32 %len, float* noalias %m, float* %dm, i32 %incm, float* noalias %n, float* %dn, i32 %incn) {
entry:
  call void (...) @__enzyme_autodiff(float (i32, float*, i32, float*, i32)* @f, i32 %len, float* noalias %m, float* %dm, i32 %incm, float* noalias %n, float* %dn, i32 %incn)
  ret void
}

define void @inactiveFirst(i32 %len, float* noalias %m, i32 %incm, float* noalias %n, float* %dn, i32 %incn) {
entry:
  call void (...) @__enzyme_autodiff(float (i32, float*, i32, float*, i32)* @f, i32 %len, metadata !"enzyme_const", float* noalias %m, i32 %incm, float* noalias %n, float* %dn, i32 %incn)
  ret void
}

define void @inactiveSecond(i32 %len, float* noalias %m, float* noalias %dm, i32 %incm, float* noalias %n, i32 %incn) {
entry:
  call void (...) @__enzyme_autodiff(float (i32, float*, i32, float*, i32)* @f, i32 %len, float* noalias %m, float* noalias %dm, i32 %incm, metadata !"enzyme_const", float* noalias %n, i32 %incn)
  ret void
}

define void @activeMod(i32 %len, float* noalias %m, float* %dm, i32 %incm, float* noalias %n, float* %dn, i32 %incn) {
entry:
  call void (...) @__enzyme_autodiff(float (i32, float*, i32, float*, i32)* @modf, i32 %len, float* noalias %m, float* %dm, i32 %incm, float* noalias %n, float* %dn, i32 %incn)
  ret void
}

define void @inactiveModFirst(i32 %len, float* noalias %m, i32 %incm, float* noalias %n, float* %dn, i32 %incn) {
entry:
  call void (...) @__enzyme_autodiff(float (i32, float*, i32, float*, i32)* @modf, i32 %len, metadata !"enzyme_const", float* noalias %m, i32 %incm, float* noalias %n, float* %dn, i32 %incn)
  ret void
}

define void @inactiveModSecond(i32 %len, float* noalias %m, float* noalias %dm, i32 %incm, float* noalias %n, i32 %incn) {
entry:
  call void (...) @__enzyme_autodiff(float (i32, float*, i32, float*, i32)* @modf, i32 %len, float* noalias %m, float* noalias %dm, i32 %incm, metadata !"enzyme_const", float* noalias %n, i32 %incn)
  ret void
}

define float @f(i32 %len, float* noalias %m, i32 %incm, float* noalias %n, i32 %incn) {
entry:
  %call = call float @cblas_sdot(i32 %len, float* %m, i32 %incm, float* %n, i32 %incn)
  ret float %call
}

define float @modf(i32 %len, float* noalias %m, i32 %incm, float* noalias %n, i32 %incn) {
entry:
  %call = call float @f(i32 %len, float* %m, i32 %incm, float* %n, i32 %incn)
  store float 0.000000e+00, float* %m
  store float 0.000000e+00, float* %n
  ret float %call
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


; CHECK: define internal void @[[active]](i32 %len, float* noalias %m, float* %"m'", i32 %incm, float* noalias %n, float* %"n'", i32 %incn, float %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @cblas_saxpy(i32 %len, float %differeturn, float* %m, i32 %incm, float* %"n'", i32 %incn)
; CHECK-NEXT:   call void @cblas_saxpy(i32 %len, float %differeturn, float* %n, i32 %incn, float* %"m'", i32 %incm)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @[[inactiveFirst]](i32 %len, float* noalias %m, i32 %incm, float* noalias %n, float* %"n'", i32 %incn, float %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @cblas_saxpy(i32 %len, float %differeturn, float* %m, i32 %incm, float* %"n'", i32 %incn)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @[[inactiveSecond]](i32 %len, float* noalias %m, float* %"m'", i32 %incm, float* noalias %n, i32 %incn, float %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @cblas_saxpy(i32 %len, float %differeturn, float* %n, i32 %incn, float* %"m'", i32 %incm)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @[[activeMod]](i32 %len, float* noalias %m, float* %"m'", i32 %incm, float* noalias %n, float* %"n'", i32 %incn, float %differeturn)
; CHECK-NEXT: entry:
; CHECK:        %call_augmented = call { float*, float* } @[[augMod:.+]](i32 %len, float* %m, float* %"m'", i32 %incm, float* %n, float* %"n'", i32 %incn)
; CHECK:        call void @[[revMod:.+]](i32 %len, float* %m, float* %"m'", i32 %incm, float* %n, float* %"n'", i32 %incn, float %differeturn, { float*, float* } %call_augmented)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal { float*, float* } @[[augMod]](i32 %len, float* noalias %m, float* %"m'", i32 %incm, float* noalias %n, float* %"n'", i32 %incn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %mallocsize = mul i32 %len, 4
; CHECK-NEXT:   %malloccall = tail call i8* @malloc(i32 %mallocsize)
; CHECK-NEXT:   %0 = bitcast i8* %malloccall to float*
; CHECK-NEXT:   %1 = icmp eq i32 %len, 0
; CHECK-NEXT:   br i1 %1, label %__enzyme_memcpy_float_32_da0sa0stride.exit, label %for.body.i

; CHECK: for.body.i:                                       ; preds = %for.body.i, %entry
; CHECK-NEXT:   %idx.i = phi i32 [ 0, %entry ], [ %idx.next.i, %for.body.i ]
; CHECK-NEXT:   %sidx.i = phi i32 [ 0, %entry ], [ %sidx.next.i, %for.body.i ]
; CHECK-NEXT:   %dst.i.i = getelementptr inbounds float, float* %0, i32 %idx.i
; CHECK-NEXT:   %src.i.i = getelementptr inbounds float, float* %m, i32 %sidx.i
; CHECK-NEXT:   %src.i.l.i = load float, float* %src.i.i
; CHECK-NEXT:   store float %src.i.l.i, float* %dst.i.i
; CHECK-NEXT:   %idx.next.i = add nuw i32 %idx.i, 1
; CHECK-NEXT:   %sidx.next.i = add nuw i32 %sidx.i, %incm
; CHECK-NEXT:   %2 = icmp eq i32 %len, %idx.next.i
; CHECK-NEXT:   br i1 %2, label %__enzyme_memcpy_float_32_da0sa0stride.exit, label %for.body.i

; CHECK: __enzyme_memcpy_float_32_da0sa0stride.exit:       ; preds = %entry, %for.body.i
; CHECK-NEXT:   %mallocsize1 = mul i32 %len, 4
; CHECK-NEXT:   %malloccall2 = tail call i8* @malloc(i32 %mallocsize1)
; CHECK-NEXT:   %3 = bitcast i8* %malloccall2 to float*
; CHECK-NEXT:   %4 = icmp eq i32 %len, 0
; CHECK-NEXT:   br i1 %4, label %__enzyme_memcpy_float_32_da0sa0stride.exit9, label %for.body.i8

; CHECK: for.body.i8:                                      ; preds = %for.body.i8, %__enzyme_memcpy_float_32_da0sa0stride.exit
; CHECK-NEXT:   %idx.i1 = phi i32 [ 0, %__enzyme_memcpy_float_32_da0sa0stride.exit ], [ %idx.next.i6, %for.body.i8 ]
; CHECK-NEXT:   %sidx.i2 = phi i32 [ 0, %__enzyme_memcpy_float_32_da0sa0stride.exit ], [ %sidx.next.i7, %for.body.i8 ]
; CHECK-NEXT:   %dst.i.i3 = getelementptr inbounds float, float* %3, i32 %idx.i1
; CHECK-NEXT:   %src.i.i4 = getelementptr inbounds float, float* %n, i32 %sidx.i2
; CHECK-NEXT:   %src.i.l.i5 = load float, float* %src.i.i4
; CHECK-NEXT:   store float %src.i.l.i5, float* %dst.i.i3
; CHECK-NEXT:   %idx.next.i6 = add nuw i32 %idx.i1, 1
; CHECK-NEXT:   %sidx.next.i7 = add nuw i32 %sidx.i2, %incn
; CHECK-NEXT:   %5 = icmp eq i32 %len, %idx.next.i6
; CHECK-NEXT:   br i1 %5, label %__enzyme_memcpy_float_32_da0sa0stride.exit9, label %for.body.i8

; CHECK: __enzyme_memcpy_float_32_da0sa0stride.exit9:      ; preds = %__enzyme_memcpy_float_32_da0sa0stride.exit, %for.body.i8
; CHECK-NEXT:   %6 = insertvalue { float*, float* } undef, float* %0, 0
; CHECK-NEXT:   %7 = insertvalue { float*, float* } %6, float* %3, 1
; CHECK-NEXT:   ret { float*, float* } %7
; CHECK-NEXT: }

; CHECK: define internal void @[[revMod]](i32 %len, float* noalias %m, float* %"m'", i32 %incm, float* noalias %n, float* %"n'", i32 %incn, float %differeturn, { float*, float* }
; CHECK-NEXT: entry:
; CHECK-NEXT:   %1 = extractvalue { float*, float* } %0, 0
; CHECK-NEXT:   %2 = extractvalue { float*, float* } %0, 1
; CHECK-NEXT:   call void @cblas_saxpy(i32 %len, float %differeturn, float* %1, i32 1, float* %"n'", i32 %incn)
; CHECK-NEXT:   call void @cblas_saxpy(i32 %len, float %differeturn, float* %2, i32 1, float* %"m'", i32 %incm)
; CHECK-NEXT:   %3 = bitcast float* %1 to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %3)
; CHECK-NEXT:   %4 = bitcast float* %2 to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %4)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @[[inactiveModFirst]](i32 %len, float* noalias %m, i32 %incm, float* noalias %n, float* %"n'", i32 %incn, float %differeturn)
; CHECK-NEXT: entry:
; CHECK:        %call_augmented = call float* @[[augModFirst:.+]](i32 %len, float* %m, i32 %incm, float* %n, float* %"n'", i32 %incn)
; CHECK:        call void @[[revModFirst:.+]](i32 %len, float* %m, i32 %incm, float* %n, float* %"n'", i32 %incn, float %differeturn, float* %call_augmented)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal float* @[[augModFirst]](i32 %len, float* noalias %m, i32 %incm, float* noalias %n, float* %"n'", i32 %incn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %mallocsize = mul i32 %len, 4
; CHECK-NEXT:   %malloccall = tail call i8* @malloc(i32 %mallocsize)
; CHECK-NEXT:   %0 = bitcast i8* %malloccall to float*
; CHECK-NEXT:   %1 = icmp eq i32 %len, 0
; CHECK-NEXT:   br i1 %1, label %__enzyme_memcpy_float_32_da0sa0stride.exit, label %for.body.i

; CHECK: for.body.i:                                       ; preds = %for.body.i, %entry
; CHECK-NEXT:   %idx.i = phi i32 [ 0, %entry ], [ %idx.next.i, %for.body.i ]
; CHECK-NEXT:   %sidx.i = phi i32 [ 0, %entry ], [ %sidx.next.i, %for.body.i ]
; CHECK-NEXT:   %dst.i.i = getelementptr inbounds float, float* %0, i32 %idx.i
; CHECK-NEXT:   %src.i.i = getelementptr inbounds float, float* %m, i32 %sidx.i
; CHECK-NEXT:   %src.i.l.i = load float, float* %src.i.i
; CHECK-NEXT:   store float %src.i.l.i, float* %dst.i.i
; CHECK-NEXT:   %idx.next.i = add nuw i32 %idx.i, 1
; CHECK-NEXT:   %sidx.next.i = add nuw i32 %sidx.i, %incm
; CHECK-NEXT:   %2 = icmp eq i32 %len, %idx.next.i
; CHECK-NEXT:   br i1 %2, label %__enzyme_memcpy_float_32_da0sa0stride.exit, label %for.body.i

; CHECK: __enzyme_memcpy_float_32_da0sa0stride.exit:       ; preds = %entry, %for.body.i
; CHECK-NEXT:   ret float* %0
; CHECK-NEXT: }

; CHECK: define internal void @[[revModFirst]](i32 %len, float* noalias %m, i32 %incm, float* noalias %n, float* %"n'", i32 %incn, float %differeturn, float*
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @cblas_saxpy(i32 %len, float %differeturn, float* %0, i32 1, float* %"n'", i32 %incn)
; CHECK-NEXT:   %1 = bitcast float* %0 to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %1)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @[[inactiveModSecond]](i32 %len, float* noalias %m, float* %"m'", i32 %incm, float* noalias %n, i32 %incn, float %differeturn)
; CHECK-NEXT: entry:
; CHECK:        %call_augmented = call float* @[[augModSecond:.+]](i32 %len, float* %m, float* %"m'", i32 %incm, float* %n, i32 %incn)
; CHECK:        call void @[[revModSecond:.+]](i32 %len, float* %m, float* %"m'", i32 %incm, float* %n, i32 %incn, float %differeturn, float* %call_augmented)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal float* @[[augModSecond]](i32 %len, float* noalias %m, float* %"m'", i32 %incm, float* noalias %n, i32 %incn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %mallocsize = mul i32 %len, 4
; CHECK-NEXT:   %malloccall = tail call i8* @malloc(i32 %mallocsize)
; CHECK-NEXT:   %0 = bitcast i8* %malloccall to float*
; CHECK-NEXT:   %1 = icmp eq i32 %len, 0
; CHECK-NEXT:   br i1 %1, label %__enzyme_memcpy_float_32_da0sa0stride.exit, label %for.body.i

; CHECK: for.body.i:                                       ; preds = %for.body.i, %entry
; CHECK-NEXT:   %idx.i = phi i32 [ 0, %entry ], [ %idx.next.i, %for.body.i ]
; CHECK-NEXT:   %sidx.i = phi i32 [ 0, %entry ], [ %sidx.next.i, %for.body.i ]
; CHECK-NEXT:   %dst.i.i = getelementptr inbounds float, float* %0, i32 %idx.i
; CHECK-NEXT:   %src.i.i = getelementptr inbounds float, float* %n, i32 %sidx.i
; CHECK-NEXT:   %src.i.l.i = load float, float* %src.i.i
; CHECK-NEXT:   store float %src.i.l.i, float* %dst.i.i
; CHECK-NEXT:   %idx.next.i = add nuw i32 %idx.i, 1
; CHECK-NEXT:   %sidx.next.i = add nuw i32 %sidx.i, %incn
; CHECK-NEXT:   %2 = icmp eq i32 %len, %idx.next.i
; CHECK-NEXT:   br i1 %2, label %__enzyme_memcpy_float_32_da0sa0stride.exit, label %for.body.i

; CHECK: __enzyme_memcpy_float_32_da0sa0stride.exit:       ; preds = %entry, %for.body.i
; CHECK-NEXT:   ret float* %0
; CHECK-NEXT: }

; CHECK: define internal void @[[revModSecond]](i32 %len, float* noalias %m, float* %"m'", i32 %incm, float* noalias %n, i32 %incn, float %differeturn, float*
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @cblas_saxpy(i32 %len, float %differeturn, float* %0, i32 1, float* %"m'", i32 %incm)
; CHECK-NEXT:   %1 = bitcast float* %0 to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %1)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

