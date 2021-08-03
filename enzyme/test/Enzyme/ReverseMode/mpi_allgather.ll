; RUN: -O2 %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

%struct.ompi_predefined_datatype_t = type opaque
%struct.ompi_predefined_communicator_t = type opaque
%struct.ompi_datatype_t = type opaque
%struct.ompi_communicator_t = type opaque

@random_datatype = external dso_local global %struct.ompi_predefined_datatype_t, align 1
@ompi_mpi_comm_world = external dso_local global %struct.ompi_predefined_communicator_t, align 1

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

; CHECK: define internal { void } @diffempi_allgather_test(double* %b, double* %"b'", i8* %recv_buf, i8* %"recv_buf'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = alloca i32, align 4
; CHECK-NEXT:   %1 = alloca i32, align 4
; CHECK-NEXT:   %"i8buf'ipc" = bitcast double* %"b'" to i8*
; CHECK-NEXT:   %i8buf = bitcast double* %b to i8*
; CHECK-NEXT:   %2 = call i32 @MPI_Allgather(i8* nonnull %i8buf, i32 1, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @random_datatype to %struct.ompi_datatype_t*), i8* %recv_buf, i32 1, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @random_datatype to %struct.ompi_datatype_t*), %struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*))
; CHECK-NEXT:   br label %invertentry

; CHECK: invertentry:                                      ; preds = %entry
; CHECK-NEXT:   %3 = call i32 @MPI_Type_size(i8* bitcast (%struct.ompi_predefined_datatype_t* @random_datatype to i8*), i32* %0)
; CHECK-NEXT:   %4 = load i32, i32* %0, align 4
; CHECK-NEXT:   %5 = zext i32 %4 to i64
; CHECK-NEXT:   %6 = mul nuw nsw i64 1, %5
; CHECK-NEXT:   %7 = tail call i8* @malloc(i64 %6)
; CHECK-NEXT:   call void @"__enzyme_mpi_sumFloat@doubleinitializer"()
; CHECK-NEXT:   %8 = load i8*, i8** @"__enzyme_mpi_sumFloat@double", align 8
; CHECK-NEXT:   %9 = call i32 @MPI_Reduce_scatter_block(i8* %"recv_buf'", i8* %7, i32 1, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @random_datatype to %struct.ompi_datatype_t*), i8* %8, %struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*))
; CHECK-NEXT:   %10 = zext i32 %4 to i64
; CHECK-NEXT:   %11 = mul nuw nsw i64 1, %10
; CHECK-NEXT:   %12 = call i32 @MPI_Comm_size(%struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*), i32* %1)
; CHECK-NEXT:   %13 = load i32, i32* %1, align 4
; CHECK-NEXT:   %14 = zext i32 %13 to i64
; CHECK-NEXT:   %15 = mul nuw nsw i64 %11, %14
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* nonnull %"recv_buf'", i8 0, i64 %15, i1 false)
; CHECK-NEXT:   %16 = bitcast i8* %7 to double*
; CHECK-NEXT:   %17 = bitcast i8* %"i8buf'ipc" to double*
; CHECK-NEXT:   %18 = udiv i64 %6, 8
; CHECK-NEXT:   call void @__enzyme_memcpyadd_doubleda1sa1(double* %16, double* %17, i64 %18)
; CHECK-NEXT:   tail call void @free(i8* nonnull %7)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
