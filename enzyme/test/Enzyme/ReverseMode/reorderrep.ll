; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -S | FileCheck %s

define void @outer(i8* %tmp71, i8* %tmp72, i8* %tmp73, i8* %tmp74) {
bb:
  call void @__enzyme_autodiff(i8* bitcast (void (double**, double*)* @matvec to i8*), i8* %tmp71, i8* %tmp72, i8* %tmp73, i8* %tmp74)
  ret void
}

declare void @__enzyme_autodiff(i8*, i8*, i8*, i8*, i8*)

define void @matvec(double** %tmp28.i.i.i.i, double* %tmp3) {
bb:
  call void @inner(double** %tmp28.i.i.i.i, double* %tmp3)
  store double* null, double** %tmp28.i.i.i.i, align 8
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define void @inner(double** %tmp35, double* %arg4) #2 {
bb:
  %tmp36 = load double*, double** %tmp35, align 8, !tbaa !2
  %tmp37 = ptrtoint double* %tmp36 to i64
  %tmp39 = icmp ne i64 %tmp37, 0
  br i1 %tmp39, label %bexit, label %bb43

bb43:                                             ; preds = %bb
  %tmp44 = udiv i64 %tmp37, 8
  br label %bexit

bexit:                                            ; preds = %bb43, %bb
  %.020 = phi i64 [ -1, %bb ], [ %tmp44, %bb43 ]
  br label %bb377

bb377:                                            ; preds = %bb381, %bexit
  %.05 = phi i64 [ %.020, %bexit ], [ %tmp441, %bb381 ]
  %tmp441 = add nsw i64 %.05, 1
  %tmp378 = icmp slt i64 %.05, 10
  br i1 %tmp378, label %bb381, label %bb450

bb381:                                            ; preds = %bb377
  %tmp384 = load double*, double** %tmp35, align 8, !tbaa !8
  %tmp389 = load double, double* %tmp384, align 8, !tbaa !10
  %ppl = fadd double %tmp389, 1.000000e+00
  store double %ppl, double* %arg4, align 8, !tbaa !10
  br label %bb377

bb450:                                            ; preds = %bb377
  ret void
}

attributes #0 = { argmemonly nounwind }
attributes #1 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.0.1-12 (tags/RELEASE_701/final)"}
!2 = !{!3, !4, i64 0}
!3 = !{!"_ZTSN5Eigen8internal16blas_data_mapperIKdlLi0ELi0EEE", !4, i64 0, !7, i64 8}
!4 = !{!"any pointer", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C++ TBAA"}
!7 = !{!"long", !5, i64 0}
!8 = !{!9, !4, i64 0}
!9 = !{!"_ZTSN5Eigen8internal16blas_data_mapperIKdlLi1ELi0EEE", !4, i64 0, !7, i64 8}
!10 = !{!11, !11, i64 0}
!11 = !{!"double", !5, i64 0}

; If a misordered extraction occurs, this can (nondeterminstically) segfault
; CHECK: define internal { double**, i64 } @augmented_inner(double** %tmp35, double** %"tmp35'", double* %arg4, double* %"arg4'")

; CHECK: define internal void @diffeinner(double** %tmp35, double** %"tmp35'", double* %arg4, double* %"arg4'", { double**, i64 } %tapeArg)
; CHECK-NEXT: bb:
; CHECK-NEXT:   %0 = extractvalue { double**, i64 } %tapeArg, 0
; CHECK-NEXT:   %tmp37 = extractvalue { double**, i64 } %tapeArg, 1
; CHECK-NEXT:   %tmp39 = icmp ne i64 %tmp37, 0
; CHECK-NEXT:   br i1 %tmp39, label %bexit, label %bb43

; CHECK: bb43:                                             ; preds = %bb
; CHECK-NEXT:   %tmp44 = udiv i64 %tmp37, 8
; CHECK-NEXT:   br label %bexit

; CHECK: bexit:                                            ; preds = %bb43, %bb
; CHECK-NEXT:   %.020 = phi i64 [ -1, %bb ], [ %tmp44, %bb43 ]
; TODO-CHECK-NEXT:   %1 = icmp sgt i64 %.020, 10
; TODO-CHECK-NEXT:   %smax = select i1 %1, i64 %.020, i64 10
; CHECK:   %[[a2:.+]] = sub i64 %smax, %.020
; CHECK-NEXT:   %[[a3:.+]] = add nuw i64 %[[a2]], 1
; CHECK-NEXT:   %mallocsize = mul nuw nsw i64 %[[a3]], 8
; CHECK-NEXT:   br label %bb377

; CHECK: bb377:                                            ; preds = %bb381, %bexit
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %bb381 ], [ 0, %bexit ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %[[a4:.+]] = add i64 %.020, %iv
; CHECK-NEXT:   %tmp378 = icmp slt i64 %[[a4]], 10
; CHECK-NEXT:   br i1 %tmp378, label %bb381, label %bb450

; CHECK: bb381:                                            ; preds = %bb377
; CHECK-NEXT:   %[[a5:.+]] = getelementptr inbounds double*, double** %0, i64 %iv
; CHECK-NEXT:   %"tmp384'il_phi" = load double*, double** %[[a5]], align 8, !invariant.group !
; CHECK-NEXT:   br label %bb377

; CHECK: bb450:                                            ; preds = %bb377
; CHECK-NEXT:   br label %invertbb450

; CHECK: invertbb:                                         ; preds = %invertbexit, %invertbb43
; CHECK-NEXT:   ret void

; CHECK: invertbb43:                                       ; preds = %invertbexit
; CHECK-NEXT:   br label %invertbb

; CHECK: invertbexit:                                      ; preds = %invertbb377
; CHECK-NEXT:   %[[a6:.+]] = bitcast double** %0 to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %[[a6]])
; CHECK-NEXT:   br i1 %tmp39, label %invertbb, label %invertbb43

; CHECK: invertbb377:                                      ; preds = %mergeinvertbb377_bb450, %invertbb381
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ %[[a2]], %mergeinvertbb377_bb450 ], [ %[[a9:.+]], %invertbb381 ]
; CHECK-NEXT:   %[[a7:.+]] = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   %[[a8:.+]] = xor i1 %[[a7]], true
; CHECK-NEXT:   br i1 %[[a7]], label %invertbexit, label %incinvertbb377

; CHECK: incinvertbb377:                                   ; preds = %invertbb377
; CHECK-NEXT:   %[[a9]] = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %invertbb381

; CHECK: invertbb381:                                      ; preds = %incinvertbb377
; CHECK-NEXT:   %[[a10:.+]] = load double, double* %"arg4'", align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"arg4'", align 8
; CHECK-NEXT:   %[[a11:.+]] = fadd fast double 0.000000e+00, %[[a10]]
; CHECK-NEXT:   %[[a12:.+]] = fadd fast double 0.000000e+00, %[[a11]]
; CHECK-NEXT:   %[[tmp37_unwrap6:.+]] = extractvalue { double**, i64 } %tapeArg, 1
; CHECK-NEXT:   %tmp39_unwrap = icmp ne i64 %[[tmp37_unwrap6]], 0
; CHECK-NEXT:   %[[tmp37_unwrap5:.+]] = extractvalue { double**, i64 } %tapeArg, 1
; CHECK-NEXT:   %tmp44_unwrap = udiv i64 %[[tmp37_unwrap5]], 8
; CHECK-NEXT:   %.020_unwrap = select i1 %tmp39_unwrap, i64 -1, i64 %tmp44_unwrap
; CHECK:   %[[_unwrap3:.+]] = sub i64 %[[smax_unwrap:.+]], %.020_unwrap
; CHECK-NEXT:   %[[a13:.+]] = add nuw i64 %[[_unwrap3]], 1
; CHECK-NEXT:   %[[a14:.+]] = extractvalue { double**, i64 } %tapeArg, 0
; CHECK-NEXT:   %[[a15:.+]] = getelementptr inbounds double*, double** %[[a14]], i64 %[[a9]]
; CHECK-NEXT:   %[[a16:.+]] = load double*, double** %[[a15]], align 8, !invariant.group !
; CHECK-NEXT:   %[[a17:.+]] = load double, double* %[[a16]], align 8
; CHECK-NEXT:   %[[a18:.+]] = fadd fast double %[[a17]], %[[a12]]
; CHECK-NEXT:   store double %[[a18]], double* %[[a16]], align 8
; CHECK-NEXT:   br label %invertbb377

; CHECK: invertbb450:                                      ; preds = %bb450
; CHECK-NEXT:   br label %mergeinvertbb377_bb450

; CHECK: mergeinvertbb377_bb450:                           ; preds = %invertbb450
; CHECK-NEXT:   br label %invertbb377
; CHECK-NEXT: }
