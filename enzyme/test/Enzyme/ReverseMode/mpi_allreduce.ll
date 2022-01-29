; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

%struct.ompi_predefined_datatype_t = type opaque
%struct.ompi_predefined_op_t = type opaque
%struct.ompi_predefined_communicator_t = type opaque
%struct.ompi_op_t = type opaque
%struct.ompi_datatype_t = type opaque
%struct.ompi_communicator_t = type opaque

@random_datatype = external dso_local global %struct.ompi_predefined_datatype_t, align 1
@ompi_mpi_op_sum = external dso_local global %struct.ompi_predefined_op_t, align 1
@ompi_mpi_comm_world = external dso_local global %struct.ompi_predefined_communicator_t, align 1

define void @mpi_allreduce_test(double* %b, i8* %global_sum_addr) {
entry:
  %i8buf = bitcast double* %b to i8*
  call i32 @MPI_Allreduce(i8* nonnull %i8buf, i8* %global_sum_addr, i32 1, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @random_datatype to %struct.ompi_datatype_t*), %struct.ompi_op_t* bitcast (%struct.ompi_predefined_op_t* @ompi_mpi_op_sum to %struct.ompi_op_t*), %struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*))
  ret void
}

declare i32 @MPI_Allreduce(i8*, i8*, i32, %struct.ompi_datatype_t*, %struct.ompi_op_t*, %struct.ompi_communicator_t*) local_unnamed_addr

; Function Attrs: nounwind uwtable
define void @caller(double* %b, double* %db, double* %sum, double* %dsum) local_unnamed_addr  {
entry:
  call void (i8*, ...) @__enzyme_autodiff(i8* bitcast (void (double*, i8*)* @mpi_allreduce_test to i8*), metadata !"enzyme_dup", double* %b, double* %db, metadata !"enzyme_dup", double* %sum, double* %dsum)
  ret void
}

declare void @__enzyme_autodiff(i8*, ...)

; CHECK: define internal void @diffempi_allreduce_test(double* %b, double* %"b'", i8* %global_sum_addr, i8* %"global_sum_addr'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = alloca i32
; CHECK-NEXT:   %i8buf = bitcast double* %b to i8*
; CHECK-NEXT:   %1 = call i32 @MPI_Allreduce(i8* nonnull %i8buf, i8* %global_sum_addr, i32 1, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @random_datatype to %struct.ompi_datatype_t*), %struct.ompi_op_t* bitcast (%struct.ompi_predefined_op_t* @ompi_mpi_op_sum to %struct.ompi_op_t*), %struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*))
; CHECK-NEXT:   %2 = call i32 @MPI_Type_size(i8* bitcast (%struct.ompi_predefined_datatype_t* @random_datatype to i8*), i32* %0)
; CHECK-NEXT:   %3 = load i32, i32* %0
; CHECK-NEXT:   %4 = zext i32 %3 to i64
; CHECK-NEXT:   %5 = tail call i8* @malloc(i64 %4)
; CHECK-NEXT:   %6 = call i32 @MPI_Allreduce(i8* %"global_sum_addr'", i8* %5, i32 1, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @random_datatype to %struct.ompi_datatype_t*), %struct.ompi_op_t* bitcast (%struct.ompi_predefined_op_t* @ompi_mpi_op_sum to %struct.ompi_op_t*), %struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*))
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* nonnull %"global_sum_addr'", i8 0, i64 %4, i1 false)
; CHECK-NEXT:   %7 = bitcast i8* %5 to double*
; CHECK-NEXT:   %8 = udiv i64 %4, 8
; CHECK-NEXT:   call void @__enzyme_memcpyadd_doubleda1sa1(double* %7, double* %"b'", i64 %8)
; CHECK-NEXT:   tail call void @free(i8* nonnull %5)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
