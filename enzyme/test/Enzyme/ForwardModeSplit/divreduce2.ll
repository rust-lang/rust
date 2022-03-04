; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -simplifycfg -early-cse-memssa -instsimplify -correlated-propagation -adce -S | FileCheck %s

; TODO optimize this style reduction

; Function Attrs: norecurse nounwind readonly uwtable
define double @alldiv(double* nocapture readonly %A, i64 %N, double %start) {
entry:
  br label %loop

loop:                                                ; preds = %9, %5
  %i = phi i64 [ 0, %entry ], [ %next, %body ]
  %reduce = phi double [ %start, %entry ], [ %div, %body ]
  %cmp = icmp ult i64 %i, %N
  br i1 %cmp, label %body, label %end

body:
  %gep = getelementptr inbounds double, double* %A, i64 %i
  %ld = load double, double* %gep, align 8, !tbaa !2
  %div = fdiv double %reduce, %ld
  %next = add nuw nsw i64 %i, 1
  br label %loop

end:                                                ; preds = %9, %3
  ret double %reduce
}

; Function Attrs: nounwind uwtable
define double @main(double* %A, double* %dA, i64 %N, double %start) {
  %r = call double (...) @__enzyme_fwdsplit(i8* bitcast (double (double*, i64, double)* @alldiv to i8*), metadata !"enzyme_nofree", double* %A, double* %dA, i64 %N, double %start, double 1.0, i8* null)
  ret double %r
}

declare double @__enzyme_fwdsplit(...)

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"Ubuntu clang version 10.0.1-++20200809072545+ef32c611aa2-1~exp1~20200809173142.193"}
!2 = !{!3, !3, i64 0}
!3 = !{!"double", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
!6 = !{!7, !7, i64 0}
!7 = !{!"any pointer", !4, i64 0}


; CHECK: define internal double @fwddiffealldiv(double* nocapture readonly %A, double* nocapture %"A'", i64 %N, double %start, double %"start'", i8* %tapeArg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = bitcast i8* %tapeArg to { double*, double* }*
; CHECK-NEXT:   %truetape = load { double*, double* }, { double*, double* }* %0
; CHECK-DAG:   %[[i1:.+]] = extractvalue { double*, double* } %truetape, 0
; CHECK-DAG:   %[[i2:.+]] = extractvalue { double*, double* } %truetape, 1
; CHECK-NEXT:   br label %loop

; CHECK: loop:                                             ; preds = %body, %entry
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %body ], [ 0, %entry ]
; CHECK-NEXT:   %"reduce'" = phi {{(fast )?}}double [ %"start'", %entry ], [ %10, %body ]
; CHECK-NEXT:   %3 = getelementptr inbounds double, double* %[[i1]], i64 %iv
; CHECK-NEXT:   %reduce = load double, double* %3, align 8,
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %cmp = icmp ne i64 %iv, %N
; CHECK-NEXT:   br i1 %cmp, label %body, label %end

; CHECK: body:                                             ; preds = %loop
; CHECK-NEXT:   %"gep'ipg" = getelementptr inbounds double, double* %"A'", i64 %iv
; CHECK-NEXT:   %4 = getelementptr inbounds double, double* %[[i2]], i64 %iv
; TODO this should keep tbaa
; CHECK-NEXT:   %ld = load double, double* %4, align 8
; CHECK-NEXT:   %5 = load double, double* %"gep'ipg", align 8
; CHECK-NEXT:   %6 = fmul fast double %"reduce'", %ld
; CHECK-NEXT:   %7 = fmul fast double %reduce, %5
; CHECK-NEXT:   %8 = fsub fast double %6, %7
; CHECK-NEXT:   %9 = fmul fast double %ld, %ld
; CHECK-NEXT:   %10 = fdiv fast double %8, %9
; CHECK-NEXT:   br label %loop

; CHECK: end:                                              ; preds = %loop
; CHECK-NEXT:   ret double %"reduce'"
; CHECK-NEXT: }
