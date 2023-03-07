; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S -enzyme-zero-cache=1 | FileCheck %s

declare dso_local double @__enzyme_autodiff(i8*, double)

define void @subsq(double addrspace(10)* writeonly nocapture %out, double %x) {
entry:
  %mul = fmul double %x, %x
  store double %mul, double addrspace(10)* %out, align 8
  ret void
}

declare {}*** @julia.get_pgcstack()
declare {} addrspace(10)* @jl_gc_alloc_typed(i8*, i64, {} addrspace(10)*)

define double @square(double %x) {
entry:
  %pg = call {}*** @julia.get_pgcstack() "enzyme_inactive" readnone "enzyme_shouldrecompute"
  %p3 = bitcast {}*** %pg to {}**
  %p4 = getelementptr inbounds {}*, {}** %p3, i64 -12
  %p5 = getelementptr inbounds {}*, {}** %p4, i64 14
  %p6 = bitcast {}** %p5 to i8**
  %p7 = load i8*, i8** %p6, align 8
  %al = call noalias nonnull dereferenceable(8) dereferenceable_or_null(8) {} addrspace(10)* @jl_gc_alloc_typed(i8* %p7, i64 8, {} addrspace(10)* addrspacecast ({}* inttoptr (i64 139806792221568 to {}*) to {} addrspace(10)*)), !enzyme_fromstack !{i64 8}
  %r = bitcast {} addrspace(10)* %al to double addrspace(10)*, !enzyme_caststack !{}
  call void @subsq(double addrspace(10)* %r, double %x)
  %ld = load double, double addrspace(10)* %r, align 8
  ret double %ld
}

define double @dsquare(double %x) local_unnamed_addr {
entry:
  %call = tail call double @__enzyme_autodiff(i8* bitcast (double (double)* @square to i8*), double %x)
  ret double %call
}

; CHECK: define internal { double } @diffesquare(double %x, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %r = alloca double, i64 1, align 8
; CHECK-NEXT:   %pg = call {}*** @julia.get_pgcstack()
; CHECK-NEXT:   %"r'ai" = alloca double, i64 1, align 8
; CHECK-NEXT:   %0 = bitcast double* %"r'ai" to {}*
; CHECK-NEXT:   %1 = bitcast {}* %0 to i8*
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* nonnull dereferenceable(8) dereferenceable_or_null(8) %1, i8 0, i64 8, i1 false)
; CHECK-NEXT:   %2 = load double, double* %"r'ai", align 8
; CHECK-NEXT:   %3 = fadd fast double %2, %differeturn
; CHECK-NEXT:   store double %3, double* %"r'ai", align 8,
; CHECK-NEXT:   %4 = addrspacecast double* %r to double addrspace(10)*
; CHECK-NEXT:   %5 = addrspacecast double* %"r'ai" to double addrspace(10)*
; CHECK-NEXT:   %6 = call { double } @diffesubsq(double addrspace(10)* %4, double addrspace(10)* %5, double %x)
; CHECK-NEXT:   ret { double } %6
; CHECK-NEXT: }
