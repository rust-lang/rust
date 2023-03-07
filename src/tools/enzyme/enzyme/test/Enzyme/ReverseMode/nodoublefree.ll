; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -simplifycfg -adce -S | FileCheck %s
; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | %lli - | FileCheck %s --check-prefix=EVAL

; EVAL: reduce_max=1.000000
; EVAL: d_reduce_max(0)=1.000000

source_filename = "multivecmax.cpp"

@.str = private unnamed_addr constant [15 x i8] c"reduce_max=%f\0A\00", align 1
@.str.1 = private unnamed_addr constant [21 x i8] c"d_reduce_max(%i)=%f\0A\00", align 1

define dso_local double @r(double %d) {
entry:
  %call5.i.i.i.i39 = call dereferenceable(8) i8* @_Znwm(i64 8)
  %tmp = bitcast i8* %call5.i.i.i.i39 to double*
  store double %d, double* %tmp, align 8
  call void @m(double* nonnull %tmp)
  ret double %d
}

define dso_local i32 @main() {
entry:
  %call = call fast double @r(double 1.000000e+00)
  %call1 = call i32 (i8*, ...) @printf(i8* nonnull dereferenceable(1) getelementptr inbounds ([15 x i8], [15 x i8]* @.str, i64 0, i64 0), double %call)
  %der = call double @_Z17__enzyme_autodiffPvPdS0_i(i8* bitcast (double (double)* @r to i8*), double 1.000000e+00)
  %call4 = call i32 (i8*, ...) @printf(i8* nonnull dereferenceable(1) getelementptr inbounds ([21 x i8], [21 x i8]* @.str.1, i64 0, i64 0), i32 0, double %der)
  ret i32 0
}

; Function Attrs: nofree nounwind
declare dso_local i32 @printf(i8* nocapture readonly, ...) 

declare dso_local double @_Z17__enzyme_autodiffPvPdS0_i(i8*, double) local_unnamed_addr

define linkonce_odr dso_local void @m(double* %__p) local_unnamed_addr {
entry:
  call void @_ZNSt16allocator_traitsISaIdEE10deallocateERS0_Pdm(double* %__p)
  ret void
}

define linkonce_odr dso_local void @_ZNSt16allocator_traitsISaIdEE10deallocateERS0_Pdm(double* %__p) local_unnamed_addr {
entry:
  %tmp = bitcast double* %__p to i8*
  call void @_ZdlPv(i8* %tmp)
  ret void
}

declare dso_local void @_ZdlPv(i8*) local_unnamed_addr

; Function Attrs: nofree
declare dso_local noalias nonnull i8* @_Znwm(i64) 

; CHECK: define internal { double } @differ(double %d, double %differeturn) 
; CHECK-NEXT: entry:
; CHECK-NEXT:   %call5.i.i.i.i39 = call noalias nonnull dereferenceable(8) dereferenceable_or_null(8) i8* @_Znwm(i64 8)
; CHECK-NEXT:   %"call5.i.i.i.i39'mi" = call noalias nonnull dereferenceable(8) dereferenceable_or_null(8) i8* @_Znwm(i64 8)
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* nonnull dereferenceable(8) dereferenceable_or_null(8) %"call5.i.i.i.i39'mi", i8 0, i64 8, i1 false)
; CHECK-NEXT:   %"tmp'ipc" = bitcast i8* %"call5.i.i.i.i39'mi" to double*
; CHECK-NEXT:   %tmp = bitcast i8* %call5.i.i.i.i39 to double*
; CHECK-NEXT:   store double %d, double* %tmp, align 8
; CHECK-NEXT:   call void @diffem(double* %tmp, double* %"tmp'ipc")
; CHECK-NEXT:   %0 = load double, double* %"tmp'ipc", align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"tmp'ipc", align 8
; CHECK-NEXT:   %1 = fadd fast double %differeturn, %0
; CHECK-NEXT:   tail call void @_ZdlPv(i8* nonnull %"call5.i.i.i.i39'mi")
; CHECK-NEXT:   tail call void @_ZdlPv(i8* nonnull %call5.i.i.i.i39)
; CHECK-NEXT:   %2 = insertvalue { double } undef, double %1, 0
; CHECK-NEXT:   ret { double } %2
; CHECK-NEXT: }

; CHECK: define internal void @diffem(double* %__p, double* %"__p'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @nofree__ZNSt16allocator_traitsISaIdEE10deallocateERS0_Pdm(double* %__p)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @nofree__ZNSt16allocator_traitsISaIdEE10deallocateERS0_Pdm(double* %__p)
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
