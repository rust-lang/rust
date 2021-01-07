; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -simplifycfg -early-cse-memssa -instsimplify -correlated-propagation -adce -S | FileCheck %s

; Function Attrs: norecurse nounwind readonly uwtable
define double @alldiv(double* nocapture readonly %A, i64 %N, double %start) {
entry:
  br label %loop

loop:                                                ; preds = %9, %5
  %i = phi i64 [ 0, %entry ], [ %next, %loop ]
  %reduce = phi double [ %start, %entry ], [ %div, %loop ]
  %gep = getelementptr inbounds double, double* %A, i64 %i
  %ld = load double, double* %gep, align 8, !tbaa !2
  %div = fdiv double %reduce, %ld
  %next = add nuw nsw i64 %i, 1
  %cmp = icmp eq i64 %next, %N
  br i1 %cmp, label %end, label %loop

end:                                                ; preds = %9, %3
  ret double %div
}

define double @alldiv2(double* nocapture readonly %A, i64 %N) {
entry:
  br label %loop

loop:                                                ; preds = %9, %5
  %i = phi i64 [ 0, %entry ], [ %next, %loop ]
  %reduce = phi double [ 2.000000e+00, %entry ], [ %div, %loop ]
  %gep = getelementptr inbounds double, double* %A, i64 %i
  %ld = load double, double* %gep, align 8, !tbaa !2
  %div = fdiv double %reduce, %ld
  %next = add nuw nsw i64 %i, 1
  %cmp = icmp eq i64 %next, %N
  br i1 %cmp, label %end, label %loop

end:                                                ; preds = %9, %3
  ret double %div
}

; Function Attrs: nounwind uwtable
define double @main(double* %A, double* %dA, i64 %N, double %start) {
  %r = call double @__enzyme_autodiff(i8* bitcast (double (double*, i64, double)* @alldiv to i8*), double* %A, double* %dA, i64 %N, double %start)
  %r2 = call double @__enzyme_autodiff2(i8* bitcast (double (double*, i64)* @alldiv2 to i8*), double* %A, double* %dA, i64 %N)
  ret double %r
}

declare double @__enzyme_autodiff(i8*, double*, double*, i64, double)
declare double @__enzyme_autodiff2(i8*, double*, double*, i64)

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

; CHECK: define internal { double } @diffealldiv(double* nocapture readonly %A, double* nocapture %"A'", i64 %N, double %start, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = add i64 %N, -1
; CHECK-NEXT:   br label %loop

; CHECK: loop:                                             ; preds = %loop, %entry
; CHECK-NEXT:   %1 = phi double [ 1.000000e+00, %entry ], [ %2, %loop ]
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %loop ], [ 0, %entry ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %gep = getelementptr inbounds double, double* %A, i64 %iv
; CHECK-NEXT:   %ld = load double, double* %gep, align 8
; CHECK-NEXT:   %2 = fmul double %1, %ld
; CHECK-NEXT:   %cmp = icmp eq i64 %iv.next, %N
; CHECK-NEXT:   br i1 %cmp, label %invertloop, label %loop

; CHECK: invertentry:                                      ; preds = %invertloop
; CHECK-NEXT:   %3 = insertvalue { double } undef, double %12, 0
; CHECK-NEXT:   ret { double } %3

; CHECK: invertloop:                                       ; preds = %loop, %incinvertloop
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ %13, %incinvertloop ], [ %0, %loop ]
; CHECK-NEXT:   %gep_unwrap = getelementptr inbounds double, double* %A, i64 %"iv'ac.0"
; CHECK-NEXT:   %ld_unwrap = load double, double* %gep_unwrap, align 8
; CHECK-NEXT:   %4 = fdiv fast double %differeturn, %2
; CHECK-NEXT:   %5 = fmul fast double %start, %4
; CHECK-NEXT:   %6 = {{(fsub fast double \-?0.000000e\+00,|fneg fast double)}} %5
; CHECK-NEXT:   %7 = fdiv fast double %6, %ld_unwrap
; CHECK-NEXT:   %"gep'ipg_unwrap" = getelementptr inbounds double, double* %"A'", i64 %"iv'ac.0"
; CHECK-NEXT:   %8 = load double, double* %"gep'ipg_unwrap", align 8
; CHECK-NEXT:   %9 = fadd fast double %8, %7
; CHECK-NEXT:   store double %9, double* %"gep'ipg_unwrap", align 8
; CHECK-NEXT:   %10 = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   %11 = fadd fast double 0.000000e+00, %4
; CHECK-NEXT:   %12 = select{{( fast)?}} i1 %10, double %11, double 0.000000e+00
; CHECK-NEXT:   br i1 %10, label %invertentry, label %incinvertloop

; CHECK: incinvertloop:                                    ; preds = %invertloop
; CHECK-NEXT:   %13 = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %invertloop
; CHECK-NEXT: }

; CHECK: define internal void @diffealldiv2(double* nocapture readonly %A, double* nocapture %"A'", i64 %N, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = add i64 %N, -1
; CHECK-NEXT:   br label %loop

; CHECK: loop:                                             ; preds = %loop, %entry
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %loop ], [ 0, %entry ]
; CHECK-NEXT:   %reduce = phi double [ 2.000000e+00, %entry ], [ %div, %loop ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %gep = getelementptr inbounds double, double* %A, i64 %iv
; CHECK-NEXT:   %ld = load double, double* %gep, align 8
; CHECK-NEXT:   %div = fdiv double %reduce, %ld
; CHECK-NEXT:   %cmp = icmp eq i64 %iv.next, %N
; CHECK-NEXT:   br i1 %cmp, label %invertloop, label %loop

; CHECK: invertentry:                                      ; preds = %invertloop
; CHECK-NEXT:   ret void

; CHECK: invertloop:                                       ; preds = %loop, %incinvertloop
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ %7, %incinvertloop ], [ %0, %loop ]
; CHECK-NEXT:   %gep_unwrap = getelementptr inbounds double, double* %A, i64 %"iv'ac.0"
; CHECK-NEXT:   %ld_unwrap = load double, double* %gep_unwrap, align 8
; CHECK-NEXT:   %1 = fmul fast double %differeturn, %div
; CHECK-NEXT:   %2 = {{(fsub fast double \-?0.000000e\+00,|fneg fast double)}} %1
; CHECK-NEXT:   %3 = fdiv fast double %2, %ld_unwrap
; CHECK-NEXT:   %"gep'ipg_unwrap" = getelementptr inbounds double, double* %"A'", i64 %"iv'ac.0"
; CHECK-NEXT:   %4 = load double, double* %"gep'ipg_unwrap", align 8
; CHECK-NEXT:   %5 = fadd fast double %4, %3
; CHECK-NEXT:   store double %5, double* %"gep'ipg_unwrap", align 8
; CHECK-NEXT:   %6 = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   br i1 %6, label %invertentry, label %incinvertloop

; CHECK: incinvertloop:                                    ; preds = %invertloop
; CHECK-NEXT:   %7 = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %invertloop
; CHECK-NEXT: }