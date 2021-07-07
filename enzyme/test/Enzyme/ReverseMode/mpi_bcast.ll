; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

%struct.ompi_predefined_datatype_t = type opaque
%struct.ompi_predefined_communicator_t = type opaque
%struct.ompi_datatype_t = type opaque
%struct.ompi_communicator_t = type opaque

@ompi_mpi_double = external dso_local global %struct.ompi_predefined_datatype_t, align 1
@ompi_mpi_comm_world = external dso_local global %struct.ompi_predefined_communicator_t, align 1

define double @mpi_bcast_test(double %b) {
entry:
  %b.addr = alloca double, align 8
  store double %b, double* %b.addr, align 8
  %0 = bitcast double* %b.addr to i8*
  %call = call i32 @MPI_Bcast(i8* nonnull %0, i32 1, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_double to %struct.ompi_datatype_t*), i32 0, %struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*))
  %1 = load double, double* %b.addr, align 8
  ret double %1
}

declare i32 @MPI_Bcast(i8*, i32, %struct.ompi_datatype_t*, i32, %struct.ompi_communicator_t*) local_unnamed_addr

; Function Attrs: nounwind uwtable
define double @caller(double %x) local_unnamed_addr  {
entry:
  %call = call double (i8*, ...) @__enzyme_autodiff(i8* bitcast (double (double)* @mpi_bcast_test to i8*), double %x)
  ret double %call
}

declare double @__enzyme_autodiff(i8*, ...)

; CHECK: @"__enzyme_mpi_sumFloat@double" = internal global i8* undef
; CHECK: @"__enzyme_mpi_sumFloat@double_initd" = internal global i1 false

; CHECK: define internal { double } @diffempi_bcast_test(double %b, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = alloca i32
; CHECK-NEXT:   %1 = alloca i32
; CHECK-NEXT:   %"b.addr'ipa" = alloca double, align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"b.addr'ipa", align 8
; CHECK-NEXT:   %b.addr = alloca double, align 8
; CHECK-NEXT:   store double %b, double* %b.addr, align 8
; CHECK-NEXT:   %"'ipc" = bitcast double* %"b.addr'ipa" to i8*
; CHECK-NEXT:   %2 = bitcast double* %b.addr to i8*
; CHECK-NEXT:   %call = call i32 @MPI_Bcast(i8* nonnull %2, i32 1, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_double to %struct.ompi_datatype_t*), i32 0, %struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*))
; CHECK-NEXT:   %3 = load double, double* %"b.addr'ipa", align 8
; CHECK-NEXT:   %4 = fadd fast double %3, %differeturn
; CHECK-NEXT:   store double %4, double* %"b.addr'ipa", align 8
; CHECK-NEXT:   %5 = call i32 @MPI_Type_size(i8* bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_double to i8*), i32* %0)
; CHECK-NEXT:   %6 = load i32, i32* %0
; CHECK-NEXT:   %7 = zext i32 %6 to i64
; CHECK-NEXT:   %8 = tail call i8* @malloc(i64 %7)
; CHECK-NEXT:   call void @"__enzyme_mpi_sumFloat@doubleinitializer"()
; CHECK-NEXT:   %9 = load i8*, i8** @"__enzyme_mpi_sumFloat@double"
; CHECK-NEXT:   %10 = call i32 (...) @MPI_Reduce(i8* %"'ipc", i8* %8, i32 1, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_double to %struct.ompi_datatype_t*), i8* %9, i32 0, %struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*))
; CHECK-NEXT:   %11 = call i32 @MPI_Comm_rank(%struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*), i32* %1)
; CHECK-NEXT:   %12 = load i32, i32* %1
; CHECK-NEXT:   %13 = icmp eq i32 %12, 0
; CHECK-NEXT:   br i1 %13, label %invertentry_root, label %invertentry_nonroot

; CHECK: invertentry_root:                                 ; preds = %entry
; CHECK-NEXT:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* %"'ipc", i8* %8, i64 %7, i1 false)
; CHECK-NEXT:   br label %invertentry_post

; CHECK: invertentry_nonroot:                              ; preds = %entry
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* nonnull %"'ipc", i8 0, i64 %7, i1 false)
; CHECK-NEXT:   br label %invertentry_post

; CHECK: invertentry_post:                                 ; preds = %invertentry_nonroot, %invertentry_root
; CHECK-NEXT:   tail call void @free(i8* nonnull %8)
; CHECK-NEXT:   %14 = load double, double* %"b.addr'ipa", align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"b.addr'ipa", align 8
; CHECK-NEXT:   %15 = insertvalue { double } undef, double %14, 0
; CHECK-NEXT:   ret { double } %15
; CHECK-NEXT: }

; CHECK: define internal void @"__enzyme_mpi_sumFloat@doubleinitializer"()
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = load i1, i1* @"__enzyme_mpi_sumFloat@double_initd"
; CHECK-NEXT:   br i1 %0, label %end, label %run

; CHECK: run:                                              ; preds = %entry
; CHECK-NEXT:   %1 = call i32 @MPI_Op_create(i8* bitcast (void (double*, double*, i32*, i8**)* @"__enzyme_mpi_sumFloat@double_run" to i8*), i32 1, i8** @"__enzyme_mpi_sumFloat@double")
; CHECK-NEXT:   store i1 true, i1* @"__enzyme_mpi_sumFloat@double_initd"
; CHECK-NEXT:   br label %end

; CHECK: end:                                              ; preds = %run, %entry
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
