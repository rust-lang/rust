; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

%struct.ompi_predefined_datatype_t = type opaque
%struct.ompi_predefined_communicator_t = type opaque
%struct.ompi_datatype_t = type opaque
%struct.ompi_communicator_t = type opaque

@random_datatype = external dso_local global %struct.ompi_predefined_datatype_t, align 4
@ompi_mpi_comm_world = external dso_local global %struct.ompi_predefined_communicator_t, align 4

define void @mpi_allgather_test(double* %b, i8* %recv_buf) {
entry:
  %i8buf = bitcast double* %b to i8*
  call i32 @MPI_Allgather(i8* nonnull %i8buf, i32 1, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @random_datatype to %struct.ompi_datatype_t*), i8* %recv_buf, i32 1, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @random_datatype to %struct.ompi_datatype_t*), %struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*))
  ret void
}

declare i32 @MPI_Allgather(i8*, i32, %struct.ompi_datatype_t*, i8*, i32, %struct.ompi_datatype_t*, %struct.ompi_communicator_t*) local_unnamed_addr

; Function Attrs: nounwind uwtable
define void @caller(double* %b, double* %db, double* %vla, double* %vla3) local_unnamed_addr  {
entry:
  call void (i8*, ...) @__enzyme_autodiff(i8* bitcast (void (double*, i8*)* @mpi_allgather_test to i8*), metadata !"enzyme_dup", double* %b, double* %db, metadata !"enzyme_dup", double* %vla, double* %vla3)
  ret void
}

declare void @__enzyme_autodiff(i8*, ...)

; CHECK: @"__enzyme_mpi_sumFloat@double" = internal global i8* undef
; CHECK: @"__enzyme_mpi_sumFloat@double_initd" = internal global i1 false

; CHECK: define internal void @diffempi_allgather_test(double* %b, double* %"b'", i8* %recv_buf, i8* %"recv_buf'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = alloca i32
; CHECK-NEXT:   %1 = alloca i32
; CHECK-NEXT:   %i8buf = bitcast double* %b to i8*
; CHECK-NEXT:   %2 = call i32 @MPI_Allgather(i8* nonnull %i8buf, i32 1, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @random_datatype to %struct.ompi_datatype_t*), i8* %recv_buf, i32 1, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @random_datatype to %struct.ompi_datatype_t*), %struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*))
; CHECK-NEXT:   %3 = call i32 @MPI_Type_size(i8* bitcast (%struct.ompi_predefined_datatype_t* @random_datatype to i8*), i32* %0)
; CHECK-NEXT:   %4 = load i32, i32* %0
; CHECK-NEXT:   %5 = zext i32 %4 to i64
; CHECK-NEXT:   %6 = tail call i8* @malloc(i64 %5)
; CHECK-NEXT:   call void @"__enzyme_mpi_sumFloat@doubleinitializer"()
; CHECK-NEXT:   %7 = load i8*, i8** @"__enzyme_mpi_sumFloat@double"
; CHECK-NEXT:   %8 = call i32 @MPI_Reduce_scatter_block(i8* %"recv_buf'", i8* %6, i32 1, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @random_datatype to %struct.ompi_datatype_t*), i8* %7, %struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*))
; CHECK-NEXT:   %9 = zext i32 %4 to i64
; CHECK-NEXT:   %10 = call i32 @MPI_Comm_size(%struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*), i32* %1)
; CHECK-NEXT:   %11 = load i32, i32* %1
; CHECK-NEXT:   %12 = zext i32 %11 to i64
; CHECK-NEXT:   %13 = mul nuw nsw i64 %9, %12
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* nonnull %"recv_buf'", i8 0, i64 %13, i1 false)
; CHECK-NEXT:   %14 = bitcast i8* %6 to double*
; CHECK-NEXT:   %15 = udiv i64 %5, 8
; CHECK-NEXT:   call void @__enzyme_memcpyadd_doubleda1sa1(double* %14, double* %"b'", i64 %15)
; CHECK-NEXT:   tail call void @free(i8* nonnull %6)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
