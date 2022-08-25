; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -adce -loop-deletion -correlated-propagation -simplifycfg -S | FileCheck %s

source_filename = "map.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare double* @_ZSt18_Rb_tree_incrementPKSt18_Rb_tree_node_base(double*) 

define double @f(double* %a2) {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %q = phi double [ %add, %for.body ], [ 0.000000e+00, %entry ]
  %iter = phi double* [ %call.i, %for.body ], [ %a2, %entry ]
  %a4 = load double, double* %iter, align 8
  %add = fadd double %q, %a4
  %call.i = tail call double* @_ZSt18_Rb_tree_incrementPKSt18_Rb_tree_node_base(double* %iter)
  %cmp.i.not = icmp eq double* %call.i, null
  br i1 %cmp.i.not, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  %q.0.lcssa = phi double [ %add, %for.body ]
  ret double %q.0.lcssa
}

define void @caller() {
entry:
  call void (...) @_Z17__enzyme_autodiffPviS_S_(i8* bitcast (double (double *)* @f to i8*), metadata !"enzyme_dup", i8* null, i8* null)
  ret void
}

declare void @_Z17__enzyme_autodiffPviS_S_(...)


; CHECK: define internal void @diffef(double* %a2, double* %"a2'", double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   br label %for.body

; CHECK: for.body:                                         ; preds = %__enzyme_exponentialallocation.exit, %entry
; CHECK-NEXT:   %_cache.0 = phi double** [ null, %entry ], [ %12, %__enzyme_exponentialallocation.exit ]
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %__enzyme_exponentialallocation.exit ], [ 0, %entry ]
; CHECK-NEXT:   %0 = phi double* [ %14, %__enzyme_exponentialallocation.exit ], [ %"a2'", %entry ]
; CHECK-NEXT:   %iter = phi double* [ %call.i, %__enzyme_exponentialallocation.exit ], [ %a2, %entry ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %1 = bitcast double** %_cache.0 to i8*
; CHECK-NEXT:   %2 = and i64 %iv.next, 1
; CHECK-NEXT:   %3 = icmp ne i64 %2, 0
; CHECK-NEXT:   %4 = call i64 @llvm.ctpop.i64(i64 %iv.next)
; CHECK-NEXT:   %5 = icmp ult i64 %4, 3
; CHECK-NEXT:   %6 = and i1 %5, %3
; CHECK-NEXT:   br i1 %6, label %grow.i, label %__enzyme_exponentialallocation.exit

; CHECK: grow.i:                                           ; preds = %for.body
; CHECK-NEXT:   %7 = call i64 @llvm.ctlz.i64(i64 %iv.next, i1 true)
; CHECK-NEXT:   %8 = sub nuw nsw i64 64, %7
; CHECK-NEXT:   %9 = shl i64 8, %8
; CHECK-NEXT:   %10 = call i8* @realloc(i8* %1, i64 %9)
; CHECK-NEXT:   br label %__enzyme_exponentialallocation.exit

; CHECK: __enzyme_exponentialallocation.exit:              ; preds = %for.body, %grow.i
; CHECK-NEXT:   %11 = phi i8* [ %10, %grow.i ], [ %1, %for.body ]
; CHECK-NEXT:   %12 = bitcast i8* %11 to double**
; CHECK-NEXT:   %13 = getelementptr inbounds double*, double** %12, i64 %iv
; CHECK-NEXT:   store double* %0, double** %13, align 8, !invariant.group !0
; CHECK-NEXT:   %14 = call double* @_ZSt18_Rb_tree_incrementPKSt18_Rb_tree_node_base(double* %0)
; CHECK-NEXT:   %call.i = tail call double* @_ZSt18_Rb_tree_incrementPKSt18_Rb_tree_node_base(double* %iter)
; CHECK-NEXT:   %cmp.i.not = icmp eq double* %call.i, null
; CHECK-NEXT:   br i1 %cmp.i.not, label %invertfor.body, label %for.body

; CHECK: invertentry:                                      ; preds = %invertfor.body
; CHECK-NEXT:   tail call void @free(i8* nonnull %11)
; CHECK-NEXT:   ret void

; CHECK: invertfor.body:                                   ; preds = %__enzyme_exponentialallocation.exit, %incinvertfor.body
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ %21, %incinvertfor.body ], [ %iv, %__enzyme_exponentialallocation.exit ]
; CHECK-NEXT:   %15 = getelementptr inbounds double*, double** %12, i64 %"iv'ac.0"
; CHECK-NEXT:   %16 = load double*, double** %15, align 8, !invariant.group !0
; CHECK-NEXT:   %17 = load double, double* %16, align 8
; CHECK-NEXT:   %18 = fadd fast double %17, %differeturn
; CHECK-NEXT:   store double %18, double* %16, align 8
; CHECK-NEXT:   %19 = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   %20 = select {{(fast )?}}i1 %19, double 0.000000e+00, double %differeturn
; CHECK-NEXT:   br i1 %19, label %invertentry, label %incinvertfor.body

; CHECK: incinvertfor.body:                                ; preds = %invertfor.body
; CHECK-NEXT:   %21 = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %invertfor.body
; CHECK-NEXT: }
