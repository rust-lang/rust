; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -adce -S | FileCheck %s
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(adce)" -enzyme-preopt=false -S | FileCheck %s

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
  %r = call double @__enzyme_fwddiff(i8* bitcast (double (double*, i64, double)* @alldiv to i8*), double* %A, double* %dA, i64 %N, double %start, double 1.0)
  ret double %r
}

declare double @__enzyme_fwddiff(i8*, double*, double*, i64, double, double)

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


; CHECK: define internal double @fwddiffealldiv(double* nocapture readonly %A, double* nocapture %"A'", i64 %N, double %start, double %"start'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   br label %loop

; CHECK: loop:                                             ; preds = %body, %entry
; CHECK-DAG:   %iv = phi i64 [ %iv.next, %body ], [ 0, %entry ]
; CHECK-DAG:   %reduce = phi double [ %start, %entry ], [ %div, %body ]
; CHECK-DAG:   %[[dreduce:.+]] = phi {{(fast )?}}double [ %"start'", %entry ], [ %[[i5:.+]], %body ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %cmp = icmp ne i64 %iv, %N
; CHECK-NEXT:   br i1 %cmp, label %body, label %end

; CHECK: body:                                             ; preds = %loop
; CHECK-NEXT:   %"gep'ipg" = getelementptr inbounds double, double* %"A'", i64 %iv
; CHECK-NEXT:   %gep = getelementptr inbounds double, double* %A, i64 %iv
; CHECK-NEXT:   %[[i0:.+]] = load double, double* %"gep'ipg"
; CHECK-NEXT:   %ld = load double, double* %gep, align 8, !tbaa !2
; CHECK-NEXT:   %div = fdiv double %reduce, %ld
; CHECK-NEXT:   %[[i1:.+]] = fmul fast double %[[dreduce]], %ld
; CHECK-NEXT:   %[[i2:.+]] = fmul fast double %reduce, %[[i0]]
; CHECK-NEXT:   %[[i3:.+]] = fsub fast double %[[i1]], %[[i2]]
; CHECK-NEXT:   %[[i4:.+]] = fmul fast double %ld, %ld
; CHECK-NEXT:   %[[i5:.+]] = fdiv fast double %[[i3]], %[[i4]]
; CHECK-NEXT:   br label %loop

; CHECK: end:                                              ; preds = %loop
; CHECK-NEXT:   ret double %[[dreduce]]
; CHECK-NEXT: }
