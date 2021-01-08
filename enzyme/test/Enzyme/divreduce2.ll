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
  %r = call double @__enzyme_autodiff(i8* bitcast (double (double*, i64, double)* @alldiv to i8*), double* %A, double* %dA, i64 %N, double %start)
  ret double %r
}

declare double @__enzyme_autodiff(i8*, double*, double*, i64, double)

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
; CHECK-NEXT:   %0 = add nuw i64 %N, 1
; CHECK-NEXT:   %mallocsize = mul nuw nsw i64 %0, 8
; CHECK-NEXT:   %malloccall = tail call noalias nonnull i8* @malloc(i64 %mallocsize)
; CHECK-NEXT:   %div_malloccache = bitcast i8* %malloccall to double*
; CHECK-NEXT:   br label %loop

; CHECK: loop:                                             ; preds = %body, %entry
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %body ], [ 0, %entry ]
; CHECK-NEXT:   %reduce = phi double [ %start, %entry ], [ %div, %body ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %cmp = icmp ne i64 %iv, %N
; CHECK-NEXT:   br i1 %cmp, label %body, label %invertloop

; CHECK: body:                                             ; preds = %loop
; CHECK-NEXT:   %gep = getelementptr inbounds double, double* %A, i64 %iv
; CHECK-NEXT:   %ld = load double, double* %gep, align 8
; CHECK-NEXT:   %div = fdiv double %reduce, %ld
; CHECK-NEXT:   %1 = getelementptr inbounds double, double* %div_malloccache, i64 %iv
; CHECK-NEXT:   store double %div, double* %1, align 8
; CHECK-NEXT:   br label %loop

; CHECK: invertentry:                                      ; preds = %invertloop
; CHECK-NEXT:   %2 = insertvalue { double } undef, double %6, 0
; CHECK-NEXT:   tail call void @free(i8* nonnull %malloccall)
; CHECK-NEXT:   ret { double } %2

; CHECK: invertloop:                                       ; preds = %loop, %incinvertloop
; CHECK-NEXT:   %"reduce'de.0" = phi double [ %d0differeduce, %incinvertloop ], [ %differeturn, %loop ]
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ %7, %incinvertloop ], [ %N, %loop ]
; CHECK-NEXT:   %3 = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   %4 = select{{( fast)?}} i1 %3, double 0.000000e+00, double %"reduce'de.0"
; CHECK-NEXT:   %5 = fadd fast double 0.000000e+00, %"reduce'de.0"
; CHECK-NEXT:   %6 = select{{( fast)?}} i1 %3, double %5, double 0.000000e+00
; CHECK-NEXT:   br i1 %3, label %invertentry, label %incinvertloop

; CHECK: incinvertloop:                                    ; preds = %invertloop
; CHECK-NEXT:   %7 = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   %gep_unwrap = getelementptr inbounds double, double* %A, i64 %7
; CHECK-NEXT:   %ld_unwrap = load double, double* %gep_unwrap, align 8
; CHECK-NEXT:   %d0differeduce = fdiv fast double %4, %ld_unwrap
; CHECK-NEXT:   %8 = getelementptr inbounds double, double* %div_malloccache, i64 %7
; CHECK-NEXT:   %9 = load double, double* %8, align 8
; CHECK-NEXT:   %10 = fmul fast double %9, %d0differeduce
; CHECK-NEXT:   %11 = {{(fsub fast double \-?0.000000e\+00,|fneg fast double)}} %10
; CHECK-NEXT:   %"gep'ipg_unwrap" = getelementptr inbounds double, double* %"A'", i64 %7
; CHECK-NEXT:   %12 = load double, double* %"gep'ipg_unwrap", align 8
; CHECK-NEXT:   %13 = fadd fast double %12, %11
; CHECK-NEXT:   store double %13, double* %"gep'ipg_unwrap", align 8
; CHECK-NEXT:   br label %invertloop
; CHECK-NEXT: }