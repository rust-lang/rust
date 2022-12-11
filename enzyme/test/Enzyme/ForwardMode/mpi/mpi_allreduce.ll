; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -adce -S | FileCheck %s
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(adce)" -enzyme-preopt=false -S | FileCheck %s

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
  call void (i8*, ...) @__enzyme_fwddiff(i8* bitcast (void (double*, i8*)* @mpi_allreduce_test to i8*), metadata !"enzyme_dup", double* %b, double* %db, metadata !"enzyme_dup", double* %sum, double* %dsum)
  ret void
}

declare void @__enzyme_fwddiff(i8*, ...)


; CHECK: define internal void @fwddiffempi_allreduce_test(double* %b, double* %"b'", i8* %global_sum_addr, i8* %"global_sum_addr'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"i8buf'ipc" = bitcast double* %"b'" to i8*
; CHECK-NEXT:   %i8buf = bitcast double* %b to i8*
; CHECK-NEXT:   %0 = call i32 @MPI_Allreduce(i8* nonnull %i8buf, i8* %global_sum_addr, i32 1, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @random_datatype to %struct.ompi_datatype_t*), %struct.ompi_op_t* bitcast (%struct.ompi_predefined_op_t* @ompi_mpi_op_sum to %struct.ompi_op_t*), %struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*))
; CHECK-NEXT:   %1 = call i32 @MPI_Allreduce(i8* %"i8buf'ipc", i8* %"global_sum_addr'", i32 1, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @random_datatype to %struct.ompi_datatype_t*), %struct.ompi_op_t* bitcast (%struct.ompi_predefined_op_t* @ompi_mpi_op_sum to %struct.ompi_op_t*), %struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*))
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
