; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

define void @caller(float* %tmp6, float* %tmp7) {
  call void (i8*, ...) @_Z17__enzyme_autodiffPvz(i8* bitcast (float (float*)* @kernel_main_wrapped to i8*), float* %tmp6, float* %tmp7)
  ret void
}

declare void @_Z17__enzyme_autodiffPvz(i8*, ...)

define float @kernel_main_wrapped(float* %tmpa) {
entry:
  %tmpcall = call float* @kernel_main(float* %tmpa)
  %r = load float, float* %tmpcall, align 4
  ret float %r
}

define float* @kernel_main(float* %tmp1) {
entry:
  %tmp11 = call i8* @malloc(i64 140)
  %tmp12 = bitcast i8* %tmp11 to float*
  %tmp13 = ptrtoint float* %tmp12 to i64
  %tmp14 = add i64 %tmp13, 127
  %tmp15 = urem i64 %tmp14, 128
  %tmp16 = sub i64 %tmp14, %tmp15
  %tmp17 = inttoptr i64 %tmp16 to float*
  %tmp29 = load float, float* %tmp1, align 4
  store float %tmp29, float* %tmp17, align 4
  ret float* %tmp12
}

declare i8* @malloc(i64)

; CHECK: define internal void @diffekernel_main_wrapped(float* %tmpa, float* %"tmpa'", float %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %tmpcall_augmented = call { { i8*, i8* }, float* } @augmented_kernel_main(float* %tmpa, float* %"tmpa'")
; CHECK-NEXT:   %subcache = extractvalue { { i8*, i8* }, float* } %tmpcall_augmented, 0
; CHECK-NEXT:   %"tmpcall'ac" = extractvalue { { i8*, i8* }, float* } %tmpcall_augmented, 1
; CHECK-NEXT:   %0 = load float, float* %"tmpcall'ac", align 4
; CHECK-NEXT:   %1 = fadd fast float %0, %differeturn
; CHECK-NEXT:   store float %1, float* %"tmpcall'ac", align 4
; CHECK-NEXT:   call void @diffekernel_main(float* %tmpa, float* %"tmpa'", { i8*, i8* } %subcache)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal { { i8*, i8* }, float* } @augmented_kernel_main(float* %tmp1, float* %"tmp1'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = alloca { { i8*, i8* }, float* }
; CHECK-NEXT:   %1 = getelementptr inbounds { { i8*, i8* }, float* }, { { i8*, i8* }, float* }* %0, i32 0, i32 0
; CHECK-NEXT:   %tmp11 = call noalias nonnull dereferenceable(140) dereferenceable_or_null(140) i8* @malloc(i64 140)
; CHECK-NEXT:   %2 = getelementptr inbounds { i8*, i8* }, { i8*, i8* }* %1, i32 0, i32 1
; CHECK-NEXT:   store i8* %tmp11, i8** %2
; CHECK-NEXT:   %"tmp11'mi" = call noalias nonnull dereferenceable(140) dereferenceable_or_null(140) i8* @malloc(i64 140)
; CHECK-NEXT:   %3 = getelementptr inbounds { i8*, i8* }, { i8*, i8* }* %1, i32 0, i32 0
; CHECK-NEXT:   store i8* %"tmp11'mi", i8** %3
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* nonnull dereferenceable(140) dereferenceable_or_null(140) %"tmp11'mi", i8 0, i64 140, i1 false)
; CHECK-NEXT:   %"tmp12'ipc" = bitcast i8* %"tmp11'mi" to float*
; CHECK-NEXT:   %tmp12 = bitcast i8* %tmp11 to float*
; CHECK-NEXT:   %tmp13 = ptrtoint float* %tmp12 to i64
; CHECK-NEXT:   %tmp14 = add i64 %tmp13, 127
; CHECK-NEXT:   %tmp15 = urem i64 %tmp14, 128
; CHECK-NEXT:   %tmp16 = sub i64 %tmp14, %tmp15
; CHECK-NEXT:   %tmp17 = inttoptr i64 %tmp16 to float*
; CHECK-NEXT:   %tmp29 = load float, float* %tmp1, align 4
; CHECK-NEXT:   store float %tmp29, float* %tmp17, align 4
; CHECK-NEXT:   %4 = getelementptr inbounds { { i8*, i8* }, float* }, { { i8*, i8* }, float* }* %0, i32 0, i32 1
; CHECK-NEXT:   store float* %"tmp12'ipc", float** %4
; CHECK-NEXT:   %5 = load { { i8*, i8* }, float* }, { { i8*, i8* }, float* }* %0
; CHECK-NEXT:   ret { { i8*, i8* }, float* } %5
; CHECK-NEXT: }

; CHECK: define internal void @diffekernel_main(float* %tmp1, float* %"tmp1'", { i8*, i8* } %tapeArg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %tmp11 = extractvalue { i8*, i8* } %tapeArg, 1
; CHECK-NEXT:   %"tmp11'mi" = extractvalue { i8*, i8* } %tapeArg, 0
; CHECK-NEXT:   %"tmp12'ipc" = bitcast i8* %"tmp11'mi" to float*
; CHECK-NEXT:   %tmp12 = bitcast i8* %tmp11 to float*
; CHECK-NEXT:   %"tmp13'ipc" = ptrtoint float* %"tmp12'ipc" to i64
; CHECK-NEXT:   %tmp13 = ptrtoint float* %tmp12 to i64
; CHECK-NEXT:   %tmp141 = add i64 %"tmp13'ipc", 127
; CHECK-NEXT:   %tmp14 = add i64 %tmp13, 127
; CHECK-NEXT:   %tmp15 = urem i64 %tmp14, 128
; CHECK-NEXT:   %tmp162 = sub i64 %tmp141, %tmp15
; CHECK-NEXT:   %"tmp17'ipc" = inttoptr i64 %tmp162 to float*
; CHECK-NEXT:   %0 = load float, float* %"tmp17'ipc", align 4
; CHECK-NEXT:   store float 0.000000e+00, float* %"tmp17'ipc", align 4
; CHECK-NEXT:   %1 = load float, float* %"tmp1'", align 4
; CHECK-NEXT:   %2 = fadd fast float %1, %0
; CHECK-NEXT:   store float %2, float* %"tmp1'", align 4
; CHECK-NEXT:   tail call void @free(i8* nonnull %"tmp11'mi")
; CHECK-NEXT:   tail call void @free(i8* %tmp11)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
