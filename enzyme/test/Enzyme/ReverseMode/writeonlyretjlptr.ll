; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S -enzyme-zero-cache=1 | FileCheck %s

declare dso_local double @__enzyme_autodiff(...)

define void @subsq({} addrspace(10)** writeonly nocapture %out, {} addrspace(10)* %x) {
entry:
  store {} addrspace(10)* %x, {} addrspace(10)** %out, align 8
  ret void
}

declare {}*** @julia.get_pgcstack()
declare {} addrspace(10)* @jl_gc_alloc_typed(i8*, i64, {} addrspace(10)*)

define double @mid({} addrspace(10)* %x) {
  %pg = call {}*** @julia.get_pgcstack() "enzyme_inactive" readnone "enzyme_shouldrecompute"
  %p3 = bitcast {}*** %pg to {}**
  %p4 = getelementptr inbounds {}*, {}** %p3, i64 -12
  %p5 = getelementptr inbounds {}*, {}** %p4, i64 14
  %p6 = bitcast {}** %p5 to i8**
  %p7 = load i8*, i8** %p6, align 8
  %al = call noalias nonnull dereferenceable(8) dereferenceable_or_null(8) {} addrspace(10)* @jl_gc_alloc_typed(i8* %p7, i64 8, {} addrspace(10)* addrspacecast ({}* inttoptr (i64 139806792221568 to {}*) to {} addrspace(10)*)), !enzyme_fromstack !{i64 8}
  %r = bitcast {} addrspace(10)* %al to {} addrspace(10)* addrspace(10)*, !enzyme_caststack !{}

  %addr = addrspacecast {} addrspace(10)* addrspace(10)* %r to {} addrspace(10)**
  call void @subsq({} addrspace(10)** %addr, {} addrspace(10)* %x)
  %l = load {} addrspace(10)*, {} addrspace(10)** %addr, align 8
  %bc = bitcast {} addrspace(10)* %l to double addrspace(10)*
  %ld = load double, double addrspace(10)* %bc
  ret double %ld
}

define double @square({} addrspace(10)* %x) {
entry:
  %m = call double @mid({} addrspace(10)* %x)
  %mul = fmul double %m, %m
  ret double %mul
}

define double @dsquare({} addrspace(10)* %x, {} addrspace(10)* %dx) {
  %call = tail call double (...) @__enzyme_autodiff(i8* bitcast (double ({} addrspace(10)*)* @square to i8*), metadata !"enzyme_dup", {} addrspace(10)* %x, {} addrspace(10)* %dx)
  ret double %call
}

; CHECK: define internal void @diffesquare({} addrspace(10)* %x, {} addrspace(10)* %"x'", double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %m_augmented = call { {} addrspace(10)*, double } @augmented_mid({} addrspace(10)* %x, {} addrspace(10)* %"x'")
; CHECK-NEXT:   %subcache = extractvalue { {} addrspace(10)*, double } %m_augmented, 0
; CHECK-NEXT:   %m = extractvalue { {} addrspace(10)*, double } %m_augmented, 1
; CHECK-NEXT:   %m0diffem = fmul fast double %differeturn, %m
; CHECK-NEXT:   %m1diffem = fmul fast double %differeturn, %m
; CHECK-NEXT:   %0 = fadd fast double %m0diffem, %m1diffem
; CHECK-NEXT:   call void @diffemid({} addrspace(10)* %x, {} addrspace(10)* %"x'", double %0, {} addrspace(10)* %subcache)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @augmented_subsq({} addrspace(10)** nocapture writeonly %out, {} addrspace(10)** nocapture %"out'", {} addrspace(10)* %x, {} addrspace(10)* %"x'") 
; CHECK-NEXT: entry:
; CHECK-NEXT:   store {} addrspace(10)* %"x'", {} addrspace(10)** %"out'", align 8
; CHECK-NEXT:   store {} addrspace(10)* %x, {} addrspace(10)** %out, align 8
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal { {} addrspace(10)*, double } @augmented_mid({} addrspace(10)* %x, {} addrspace(10)* %"x'")
; CHECK-NEXT:   %1 = alloca { {} addrspace(10)*, double }
; CHECK-NEXT:   %2 = getelementptr inbounds { {} addrspace(10)*, double }, { {} addrspace(10)*, double }* %1, i32 0, i32 0
; CHECK-NEXT:   store {} addrspace(10)* null, {} addrspace(10)** %2
; CHECK-NEXT:   %r = alloca {} addrspace(10)*, i64 1, align 8
; CHECK-NEXT:   %pg = call {}*** @julia.get_pgcstack()
; CHECK-NEXT:   %p3 = bitcast {}*** %pg to {}**
; CHECK-NEXT:   %p4 = getelementptr inbounds {}*, {}** %p3, i64 -12
; CHECK-NEXT:   %p5 = getelementptr inbounds {}*, {}** %p4, i64 14
; CHECK-NEXT:   %p6 = bitcast {}** %p5 to i8**
; CHECK-NEXT:   %p7 = load i8*, i8** %p6, align 8
; CHECK-NEXT:   %"al'mi" = call noalias nonnull dereferenceable(8) dereferenceable_or_null(8) {} addrspace(10)* @jl_gc_alloc_typed(i8* %p7, i64 8, {} addrspace(10)* addrspacecast ({}* inttoptr (i64 139806792221568 to {}*) to {} addrspace(10)*))
; CHECK-NEXT:   store {} addrspace(10)* %"al'mi", {} addrspace(10)** %2
; CHECK-NEXT:   %3 = bitcast {} addrspace(10)* %"al'mi" to i8 addrspace(10)*
; CHECK-NEXT:   call void @llvm.memset.p10i8.i64(i8 addrspace(10)* nonnull dereferenceable(8) dereferenceable_or_null(8) %3, i8 0, i64 8, i1 false)
; CHECK-NEXT:   %"r'ipc" = bitcast {} addrspace(10)* %"al'mi" to {} addrspace(10)* addrspace(10)*
; CHECK-NEXT:   %"addr'ipc" = addrspacecast {} addrspace(10)* addrspace(10)* %"r'ipc" to {} addrspace(10)**
; CHECK-NEXT:   call void @augmented_subsq({} addrspace(10)** %r, {} addrspace(10)** %"addr'ipc", {} addrspace(10)* %x, {} addrspace(10)* %"x'")
; CHECK-NEXT:   %l = load {} addrspace(10)*, {} addrspace(10)** %r, align 8
; CHECK-NEXT:   %bc = bitcast {} addrspace(10)* %l to double addrspace(10)*
; CHECK-NEXT:   %ld = load double, double addrspace(10)* %bc
; CHECK-NEXT:   %4 = getelementptr inbounds { {} addrspace(10)*, double }, { {} addrspace(10)*, double }* %1, i32 0, i32 1
; CHECK-NEXT:   store double %ld, double* %4
; CHECK-NEXT:   %5 = load { {} addrspace(10)*, double }, { {} addrspace(10)*, double }* %1
; CHECK-NEXT:   ret { {} addrspace(10)*, double } %5
; CHECK-NEXT: }

; CHECK: define internal void @diffemid({} addrspace(10)* %x, {} addrspace(10)* %"x'", double %differeturn, {} addrspace(10)* %"al'mi")
; CHECK-NEXT: invert:
; CHECK-NEXT:   %pg = call {}*** @julia.get_pgcstack()
; CHECK-NEXT:   %"r'ipc" = bitcast {} addrspace(10)* %"al'mi" to {} addrspace(10)* addrspace(10)*
; CHECK-NEXT:   %"addr'ipc" = addrspacecast {} addrspace(10)* addrspace(10)* %"r'ipc" to {} addrspace(10)**
; CHECK-NEXT:   %"l'ipl" = load {} addrspace(10)*, {} addrspace(10)** %"addr'ipc", align 8
; CHECK-NEXT:   %"bc'ipc" = bitcast {} addrspace(10)* %"l'ipl" to double addrspace(10)*
; CHECK-NEXT:   %0 = load double, double addrspace(10)* %"bc'ipc"
; CHECK-NEXT:   %1 = fadd fast double %0, %differeturn
; CHECK-NEXT:   store double %1, double addrspace(10)* %"bc'ipc"
; CHECK-NEXT:   call void @diffesubsq({} addrspace(10)** null, {} addrspace(10)** null, {} addrspace(10)* %x, {} addrspace(10)* %"x'")
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @diffesubsq({} addrspace(10)** nocapture writeonly %out, {} addrspace(10)** nocapture %"out'", {} addrspace(10)* %x, {} addrspace(10)* %"x'") 
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
