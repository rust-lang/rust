; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -adce -loop-deletion -correlated-propagation -adce -simplifycfg -S | FileCheck %s
source_filename = "/mnt/pci4/wmdata/Enzyme2/enzyme/test/Integration/ReverseMode/eigensumsqdyn.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare double @__enzyme_autodiff(i8*, double*, double*)

define void @caller(double* %ptmp8, double* %dtmp8) {
  %call = call double @__enzyme_autodiff(i8* bitcast (double (double*)* @_ZL6matvecPKN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EEES3_ to i8*), double* %ptmp8, double* %dtmp8)
  ret void
}

define internal double @_ZL6matvecPKN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EEES3_(double* noalias %W) {
entry:
  %call = tail call noalias i8* @malloc(i64 8) #12
  %tmp1 = bitcast i8* %call to double*
  br label %for.body.i.i.i.i.i.i.i

for.body.i.i.i.i.i.i.i:                           ; preds = %for.body.i.i.i.i.i.i.i, %entry
  %tiv138 = phi i64 [ %tiv.next139, %for.body.i.i.i.i.i.i.i ], [ 0, %entry ]
  %tiv.next139 = add nuw nsw i64 %tiv138, 1
  %arrayidxOut = getelementptr inbounds double, double* %tmp1, i64 %tiv138
  %arrayidx.i6.i.i.i.i.i.i.i.i.i = getelementptr inbounds double, double* %W, i64 %tiv138
  %tmp24 = load double, double* %arrayidx.i6.i.i.i.i.i.i.i.i.i, align 8, !tbaa !9
  store double %tmp24, double* %arrayidxOut, align 8, !tbaa !9
  %inc.i.i.i.i.i.i.i = add nuw nsw i64 %tiv138, 1
  %exitcond.i.i.i.i.i.i.i = icmp eq i64 %inc.i.i.i.i.i.i.i, 16
  br i1 %exitcond.i.i.i.i.i.i.i, label %mid, label %for.body.i.i.i.i.i.i.i

mid:                                              ; preds = %for.body.i.i.i.i.i.i.i
  br label %for.body

for.body:                           ; preds = %for.body.i.i.i.i.i.i.i, %entry
  %tiv = phi i64 [ %tiv.next, %for.body ], [ 0, %mid ]
  %grape = phi double [ %add, %for.body ], [ 0.000000e+00, %mid ]
  %tiv.next = add nuw nsw i64 %tiv, 1
  %arrayidxOut2 = getelementptr inbounds double, double* %tmp1, i64 %tiv138
  %tmpld = load double, double* %arrayidxOut2, align 8, !tbaa !9
  %add = fadd double %grape, %tmpld
  %inc = add nuw nsw i64 %tiv, 1
  %exitcond = icmp eq i64 %inc, 16
  br i1 %exitcond, label %exit, label %for.body

exit:                                             ; preds = %for.cond10.for.cond.cleanup13_crit_edge.us.i.i.i
  %tmp13 = bitcast double* %tmp1 to i8*
  call void @free(i8* %tmp13)
  ret double %add
}

declare dso_local void @free(i8* nocapture)

; Function Attrs: nofree nounwind
declare dso_local noalias i8* @malloc(i64)

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 10.0.1 (git@github.com:llvm/llvm-project ef32c611aa214dea855364efd7ba451ec5ec3f74)"}
!2 = !{!3, !4, i64 0}
!3 = !{!"_ZTSN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EEE", !4, i64 0, !7, i64 8, !7, i64 16}
!4 = !{!"any pointer", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C++ TBAA"}
!7 = !{!"long", !5, i64 0}
!8 = !{!7, !7, i64 0}
!9 = !{!10, !10, i64 0}
!10 = !{!"double", !5, i64 0}
!11 = distinct !{!11, !12}
!12 = !{!"llvm.loop.isvectorized", i32 1}
!13 = distinct !{!13, !12}
!14 = !{!3, !7, i64 8}
!15 = !{!4, !4, i64 0}
!16 = !{!17}
!17 = distinct !{!17, !18}
!18 = distinct !{!18, !"LVerDomain"}
!19 = !{!3, !7, i64 16}
!20 = !{!21, !7, i64 32}
!21 = !{!"_ZTSN5Eigen8internal15level3_blockingIddEE", !4, i64 0, !4, i64 8, !7, i64 16, !7, i64 24, !7, i64 32}
!22 = !{!21, !7, i64 16}
!23 = !{!24, !7, i64 40}
!24 = !{!"_ZTSN5Eigen8internal19gemm_blocking_spaceILi0EddLin1ELin1ELin1ELi1ELb0EEE", !7, i64 40, !7, i64 48}
!25 = !{!21, !7, i64 24}
!26 = !{!24, !7, i64 48}
!27 = !{!21, !4, i64 0}
!28 = !{!21, !4, i64 8}
!29 = !{!"branch_weights", i32 1, i32 1048575}
!30 = !{!31, !31, i64 0}
!31 = !{!"int", !5, i64 0}
!32 = !{!33, !7, i64 0}
!33 = !{!"_ZTSN5Eigen8internal10CacheSizesE", !7, i64 0, !7, i64 8, !7, i64 16}
!34 = !{!33, !7, i64 8}
!35 = !{!33, !7, i64 16}
!36 = !{i32 -2143964507}
!37 = !{i32 -2143965119}
!38 = !{i32 -2143964813}
!39 = !{i32 -2143964660}
!40 = !{i32 -2143964966}
!41 = !{!5, !5, i64 0}
!42 = !{i32 -2142602162}
!43 = !{i32 -2142601457}
!44 = !{!45, !4, i64 0}
!45 = !{!"_ZTSN5Eigen8internal16blas_data_mapperIKdlLi0ELi0EEE", !4, i64 0, !7, i64 8}
!46 = !{!45, !7, i64 8}
!47 = !{!48}
!48 = distinct !{!48, !49}
!49 = distinct !{!49, !"LVerDomain"}
!50 = !{!51}
!51 = distinct !{!51, !49}
!52 = distinct !{!52, !12}
!53 = distinct !{!53, !12}
!54 = !{!55, !4, i64 0}
!55 = !{!"_ZTSN5Eigen8internal16blas_data_mapperIdlLi0ELi0EEE", !4, i64 0, !7, i64 8}
!56 = !{!55, !7, i64 8}
!57 = !{i32 -2142602583}
!58 = !{i32 -2142602529}
!59 = !{i32 -2142602466}
!60 = !{i32 -2142609389}
!61 = !{i32 -2142608747}
!62 = !{i32 -2142608693}
!63 = !{i32 -2142608630}
!64 = !{i32 -2142607982}
!65 = !{i32 -2142607928}
!66 = !{i32 -2142607865}
!67 = !{i32 -2142607217}
!68 = !{i32 -2142607163}
!69 = !{i32 -2142607100}
!70 = !{i32 -2142606452}
!71 = !{i32 -2142606398}
!72 = !{i32 -2142606335}
!73 = !{i32 -2142605687}
!74 = !{i32 -2142605633}
!75 = !{i32 -2142605570}
!76 = !{i32 -2142604922}
!77 = !{i32 -2142604868}
!78 = !{i32 -2142604805}
!79 = !{i32 -2142604157}
!80 = !{i32 -2142604103}
!81 = !{i32 -2142604040}
!82 = !{i32 -2142603392}
!83 = !{i32 -2142603338}
!84 = !{i32 -2142603275}
!85 = !{i32 -2142603223}
!86 = !{i32 -2142618569}
!87 = !{i32 -2142617682}
!88 = !{i32 -2142617628}
!89 = !{i32 -2142617565}
!90 = !{i32 -2142616672}
!91 = !{i32 -2142616618}
!92 = !{i32 -2142616555}
!93 = !{i32 -2142615662}
!94 = !{i32 -2142615608}
!95 = !{i32 -2142615545}
!96 = !{i32 -2142614652}
!97 = !{i32 -2142614598}
!98 = !{i32 -2142614535}
!99 = !{i32 -2142613642}
!100 = !{i32 -2142613588}
!101 = !{i32 -2142613525}
!102 = !{i32 -2142612632}
!103 = !{i32 -2142612578}
!104 = !{i32 -2142612515}
!105 = !{i32 -2142611622}
!106 = !{i32 -2142611568}
!107 = !{i32 -2142611505}
!108 = !{i32 -2142610612}
!109 = !{i32 -2142610558}
!110 = !{i32 -2142610495}
!111 = !{i32 -2142610443}
!112 = !{i32 -2142609558}
!113 = !{i32 -2142609504}
!114 = !{i32 -2142609441}

; TODO completely eliminate malloc for forward since all previous loads from are unnecessary
; CHECK: define internal void @diffe_ZL6matvecPKN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EEES3_(double* noalias %W, double* %"W'", double %differeturn) 
; CHECK-NEXT: entry:
; CHECK-NEXT:   %call = tail call noalias nonnull dereferenceable(8) dereferenceable_or_null(8) i8* @malloc(i64 8)
; CHECK-NEXT:   %"call'mi" = tail call noalias nonnull dereferenceable(8) dereferenceable_or_null(8) i8* @malloc(i64 8)
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* nonnull dereferenceable(8) dereferenceable_or_null(8) %"call'mi", i8 0, i64 8, i1 false)
; CHECK-NEXT:   %"tmp1'ipc" = bitcast i8* %"call'mi" to double*
; CHECK-NEXT:   %tmp1 = bitcast i8* %call to double*
; CHECK-NEXT:   br label %for.body.i.i.i.i.i.i.i

; CHECK: for.body.i.i.i.i.i.i.i:                           ; preds = %for.body.i.i.i.i.i.i.i, %entry
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %for.body.i.i.i.i.i.i.i ], [ 0, %entry ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %arrayidxOut = getelementptr inbounds double, double* %tmp1, i64 %iv
; CHECK-NEXT:   %arrayidx.i6.i.i.i.i.i.i.i.i.i = getelementptr inbounds double, double* %W, i64 %iv
; CHECK-NEXT:   %tmp24 = load double, double* %arrayidx.i6.i.i.i.i.i.i.i.i.i, align 8, !tbaa !2
; CHECK-NEXT:   store double %tmp24, double* %arrayidxOut, align 8, !tbaa !2
; CHECK-NEXT:   %exitcond.i.i.i.i.i.i.i = icmp eq i64 %iv.next, 16
; CHECK-NEXT:   br i1 %exitcond.i.i.i.i.i.i.i, label %exit, label %for.body.i.i.i.i.i.i.i

; CHECK: exit:                                             ; preds = %for.body.i.i.i.i.i.i.i
; CHECK-NEXT:   call void @free(i8* nonnull %call)
; CHECK-NEXT:   br label %invertfor.body

; CHECK: invertentry:                                      ; preds = %invertfor.body.i.i.i.i.i.i.i
; CHECK-NEXT:   tail call void @free(i8* nonnull %"call'mi")
; CHECK-NEXT:   ret void

; CHECK: invertfor.body.i.i.i.i.i.i.i:                     ; preds = %invertfor.body, %incinvertfor.body.i.i.i.i.i.i.i
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ %4, %incinvertfor.body.i.i.i.i.i.i.i ], [ 15, %invertfor.body ]
; CHECK-NEXT:   %"arrayidxOut'ipg_unwrap" = getelementptr inbounds double, double* %"tmp1'ipc", i64 %"iv'ac.0"
; CHECK-NEXT:   %0 = load double, double* %"arrayidxOut'ipg_unwrap", align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"arrayidxOut'ipg_unwrap", align 8
; CHECK-NEXT:   %"arrayidx.i6.i.i.i.i.i.i.i.i.i'ipg_unwrap" = getelementptr inbounds double, double* %"W'", i64 %"iv'ac.0"
; CHECK-NEXT:   %1 = load double, double* %"arrayidx.i6.i.i.i.i.i.i.i.i.i'ipg_unwrap", align 8
; CHECK-NEXT:   %2 = fadd fast double %1, %0
; CHECK-NEXT:   store double %2, double* %"arrayidx.i6.i.i.i.i.i.i.i.i.i'ipg_unwrap", align 8
; CHECK-NEXT:   %3 = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   br i1 %3, label %invertentry, label %incinvertfor.body.i.i.i.i.i.i.i

; CHECK: incinvertfor.body.i.i.i.i.i.i.i:                  ; preds = %invertfor.body.i.i.i.i.i.i.i
; CHECK-NEXT:   %4 = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %invertfor.body.i.i.i.i.i.i.i

; CHECK: invertfor.body:                                   ; preds = %exit, %incinvertfor.body
; CHECK-NEXT:   %"iv1'ac.0" = phi i64 [ 15, %exit ], [ %8, %incinvertfor.body ]
; CHECK-NEXT:   %"arrayidxOut2'ipg_unwrap" = getelementptr inbounds double, double* %"tmp1'ipc", i64 15
; CHECK-NEXT:   %5 = load double, double* %"arrayidxOut2'ipg_unwrap", align 8
; CHECK-NEXT:   %6 = fadd fast double %5, %differeturn
; CHECK-NEXT:   store double %6, double* %"arrayidxOut2'ipg_unwrap", align 8
; CHECK-NEXT:   %7 = icmp eq i64 %"iv1'ac.0", 0
; CHECK-NEXT:   br i1 %7, label %invertfor.body.i.i.i.i.i.i.i, label %incinvertfor.body

; CHECK: incinvertfor.body:                                ; preds = %invertfor.body
; CHECK-NEXT:   %8 = add nsw i64 %"iv1'ac.0", -1

