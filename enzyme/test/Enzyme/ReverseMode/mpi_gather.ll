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
; CHECK-NEXT:   %0 = alloca i32, align 4
; CHECK-NEXT:   %1 = alloca i32, align 4
; CHECK-NEXT:   %2 = alloca i32, align 4
; CHECK-NEXT:   %"i8buf'ipc" = bitcast double* %"b'" to i8*
; CHECK-NEXT:   %i8buf = bitcast double* %b to i8*
; CHECK-NEXT:   %3 = call i32 @MPI_Gather(i8* nonnull %i8buf, i32 1, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @random_datatype to %struct.ompi_datatype_t*), i8* %recv_buf, i32 1, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @random_datatype to %struct.ompi_datatype_t*), i32 0, %struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*))
; CHECK-NEXT:   br label %invertentry

; CHECK: invertentry:                                      ; preds = %entry
; CHECK-NEXT:   %4 = call i32 @MPI_Comm_rank(%struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*), i32* %0)
; CHECK-NEXT:   %5 = load i32, i32* %0, align 4
; CHECK-NEXT:   %6 = call i32 @MPI_Type_size(i8* bitcast (%struct.ompi_predefined_datatype_t* @random_datatype to i8*), i32* %1)
; CHECK-NEXT:   %7 = load i32, i32* %1, align 4
; CHECK-NEXT:   %8 = zext i32 %7 to i64
; CHECK-NEXT:   %9 = mul nuw nsw i64 1, %8
; CHECK-NEXT:   %10 = tail call i8* @malloc(i64 %9)
; CHECK-NEXT:   %11 = call i32 @MPI_Scatter(i8* %"recv_buf'", i32 1, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @random_datatype to %struct.ompi_datatype_t*), i8* %10, i32 1, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @random_datatype to %struct.ompi_datatype_t*), i32 0, %struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*))
; CHECK-NEXT:   %12 = icmp eq i32 %5, 0
; CHECK-NEXT:   br i1 %12, label %invertentry_root, label %invertentry_post

; CHECK: invertentry_root:                                 ; preds = %invertentry
; CHECK-NEXT:   %13 = zext i32 %7 to i64
; CHECK-NEXT:   %14 = mul nuw nsw i64 1, %13
; CHECK-NEXT:   %15 = call i32 @MPI_Comm_size(%struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*), i32* %2)
; CHECK-NEXT:   %16 = load i32, i32* %2, align 4
; CHECK-NEXT:   %17 = zext i32 %16 to i64
; CHECK-NEXT:   %18 = mul nuw nsw i64 %14, %17
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* nonnull %"recv_buf'", i8 0, i64 %18, i1 false)
; CHECK-NEXT:   br label %invertentry_post

; CHECK: invertentry_post:                                 ; preds = %invertentry_root, %invertentry
; CHECK-NEXT:   %19 = bitcast i8* %10 to double*
; CHECK-NEXT:   %20 = bitcast i8* %"i8buf'ipc" to double*
; CHECK-NEXT:   %21 = udiv i64 %9, 8
; CHECK-NEXT:   call void @__enzyme_memcpyadd_doubleda1sa1(double* %19, double* %20, i64 %21)
; CHECK-NEXT:   tail call void @free(i8* nonnull %10)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }