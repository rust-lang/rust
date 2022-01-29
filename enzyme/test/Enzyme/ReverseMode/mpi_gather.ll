; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

%struct.ompi_predefined_datatype_t = type opaque
%struct.ompi_predefined_communicator_t = type opaque
%struct.ompi_datatype_t = type opaque
%struct.ompi_communicator_t = type opaque

@random_datatype = external dso_local global %struct.ompi_predefined_datatype_t, align 1
@ompi_mpi_comm_world = external dso_local global %struct.ompi_predefined_communicator_t, align 1

define void @mpi_gather_test(double* %b, i8* %recv_buf) {
entry:
  %i8buf = bitcast double* %b to i8*
  call i32 @MPI_Gather(i8* nonnull %i8buf, i32 1, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @random_datatype to %struct.ompi_datatype_t*), i8* %recv_buf, i32 1, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @random_datatype to %struct.ompi_datatype_t*), i32 0, %struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*))
  ret void
}

declare i32 @MPI_Gather(i8*, i32, %struct.ompi_datatype_t*, i8*, i32, %struct.ompi_datatype_t*, i32, %struct.ompi_communicator_t*) local_unnamed_addr

; Function Attrs: nounwind uwtable
define void @caller(double* %b, double* %db, double* %vla, double* %vla3) local_unnamed_addr  {
entry:
  call void (i8*, ...) @__enzyme_autodiff(i8* bitcast (void (double*, i8*)* @mpi_gather_test to i8*), metadata !"enzyme_dup", double* %b, double* %db, metadata !"enzyme_dup", double* %vla, double* %vla3)
  ret void
}

declare void @__enzyme_autodiff(i8*, ...)

; CHECK: define internal void @diffempi_gather_test(double* %b, double* %"b'", i8* %recv_buf, i8* %"recv_buf'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = alloca i32
; CHECK-NEXT:   %1 = alloca i32
; CHECK-NEXT:   %2 = alloca i32
; CHECK-NEXT:   %i8buf = bitcast double* %b to i8*
; CHECK-NEXT:   %3 = call i32 @MPI_Gather(i8* nonnull %i8buf, i32 1, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @random_datatype to %struct.ompi_datatype_t*), i8* %recv_buf, i32 1, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @random_datatype to %struct.ompi_datatype_t*), i32 0, %struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*))
; CHECK-NEXT:   %4 = call i32 @MPI_Comm_rank(%struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*), i32* %0)
; CHECK-NEXT:   %5 = load i32, i32* %0
; CHECK-NEXT:   %6 = call i32 @MPI_Type_size(i8* bitcast (%struct.ompi_predefined_datatype_t* @random_datatype to i8*), i32* %1)
; CHECK-NEXT:   %7 = load i32, i32* %1
; CHECK-NEXT:   %8 = zext i32 %7 to i64
; CHECK-NEXT:   %9 = tail call i8* @malloc(i64 %8)
; CHECK-NEXT:   %10 = call i32 @MPI_Scatter(i8* %"recv_buf'", i32 1, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @random_datatype to %struct.ompi_datatype_t*), i8* %9, i32 1, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @random_datatype to %struct.ompi_datatype_t*), i32 0, %struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*))
; CHECK-NEXT:   %11 = icmp eq i32 %5, 0
; CHECK-NEXT:   br i1 %11, label %invertentry_root, label %invertentry_post

; CHECK: invertentry_root:                                 ; preds = %entry
; CHECK-NEXT:   %12 = zext i32 %7 to i64
; CHECK-NEXT:   %13 = call i32 @MPI_Comm_size(%struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*), i32* %2)
; CHECK-NEXT:   %14 = load i32, i32* %2
; CHECK-NEXT:   %15 = zext i32 %14 to i64
; CHECK-NEXT:   %16 = mul nuw nsw i64 %12, %15
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* nonnull %"recv_buf'", i8 0, i64 %16, i1 false)
; CHECK-NEXT:   br label %invertentry_post

; CHECK: invertentry_post:                                 ; preds = %invertentry_root, %entry
; CHECK-NEXT:   %17 = bitcast i8* %9 to double*
; CHECK-NEXT:   %18 = udiv i64 %8, 8
; CHECK-NEXT:   call void @__enzyme_memcpyadd_doubleda1sa1(double* %17, double* %"b'", i64 %18)
; CHECK-NEXT:   tail call void @free(i8* nonnull %9)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
