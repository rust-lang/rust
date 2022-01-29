; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

%struct.ompi_predefined_datatype_t = type opaque
%struct.ompi_predefined_communicator_t = type opaque
%struct.ompi_datatype_t = type opaque
%struct.ompi_communicator_t = type opaque

@random_datatype = external dso_local global %struct.ompi_predefined_datatype_t, align 1
@ompi_mpi_comm_world = external dso_local global %struct.ompi_predefined_communicator_t, align 1

define void @mpi_scatter_test(double* %send_buf, i8* %recv_buf) {
entry:
  %i8buf = bitcast double* %send_buf to i8*
  %call = call i32 @MPI_Scatter(i8* nonnull %i8buf, i32 1, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @random_datatype to %struct.ompi_datatype_t*), i8* %recv_buf, i32 1, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @random_datatype to %struct.ompi_datatype_t*), i32 0, %struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*))
  ret void
}

declare i32 @MPI_Scatter(i8*, i32, %struct.ompi_datatype_t*, i8*, i32, %struct.ompi_datatype_t*, i32, %struct.ompi_communicator_t*) local_unnamed_addr

; Function Attrs: nounwind uwtable
define void @caller(double* %vla, double* %vla3, double* %b, double* %db) local_unnamed_addr  {
entry:
  call void (i8*, ...) @__enzyme_autodiff(i8* bitcast (void (double*, i8*)* @mpi_scatter_test to i8*), metadata !"enzyme_dup", double* %vla, double* %vla3, metadata !"enzyme_dup", double* %b, double* %db)
  ret void
}

declare void @__enzyme_autodiff(i8*, ...)

; CHECK: define internal void @diffempi_scatter_test(double* %send_buf, double* %"send_buf'", i8* %recv_buf, i8* %"recv_buf'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = alloca i32
; CHECK-NEXT:   %1 = alloca i32
; CHECK-NEXT:   %2 = alloca i32
; CHECK-NEXT:   %i8buf = bitcast double* %send_buf to i8*
; CHECK-NEXT:   %call = call i32 @MPI_Scatter(i8* nonnull %i8buf, i32 1, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @random_datatype to %struct.ompi_datatype_t*), i8* %recv_buf, i32 1, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @random_datatype to %struct.ompi_datatype_t*), i32 0, %struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*))
; CHECK-NEXT:   %3 = call i32 @MPI_Comm_rank(%struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*), i32* %0)
; CHECK-NEXT:   %4 = load i32, i32* %0
; CHECK-NEXT:   %5 = call i32 @MPI_Type_size(i8* bitcast (%struct.ompi_predefined_datatype_t* @random_datatype to i8*), i32* %1)
; CHECK-NEXT:   %6 = load i32, i32* %1
; CHECK-NEXT:   %7 = zext i32 %6 to i64
; CHECK-NEXT:   %8 = icmp eq i32 %4, 0
; CHECK-NEXT:   br i1 %8, label %invertentry_root, label %invertentry_post

; CHECK: invertentry_root:                                 ; preds = %entry
; CHECK-NEXT:   %9 = zext i32 %6 to i64
; CHECK-NEXT:   %10 = call i32 @MPI_Comm_size(%struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*), i32* %2)
; CHECK-NEXT:   %11 = load i32, i32* %2
; CHECK-NEXT:   %12 = zext i32 %11 to i64
; CHECK-NEXT:   %13 = mul nuw nsw i64 %9, %12
; CHECK-NEXT:   %14 = tail call i8* @malloc(i64 %13)
; CHECK-NEXT:   br label %invertentry_post

; CHECK: invertentry_post:                                 ; preds = %invertentry_root, %entry
; CHECK-NEXT:   %15 = phi i8* [ %14, %invertentry_root ], [ undef, %entry ]
; CHECK-NEXT:   %16 = phi i64 [ %13, %invertentry_root ], [ undef, %entry ]
; CHECK-NEXT:   %17 = call i32 @MPI_Gather(i8* %"recv_buf'", i32 1, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @random_datatype to %struct.ompi_datatype_t*), i8* %15, i32 1, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @random_datatype to %struct.ompi_datatype_t*), i32 0, %struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*))
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* nonnull %"recv_buf'", i8 0, i64 %7, i1 false)
; CHECK-NEXT:   %18 = icmp eq i32 %4, 0
; CHECK-NEXT:   br i1 %18, label %invertentry_post_root, label %invertentry_post_post

; CHECK: invertentry_post_root:                            ; preds = %invertentry_post
; CHECK-NEXT:   %19 = bitcast i8* %15 to double*
; CHECK-NEXT:   %20 = udiv i64 %16, 8
; CHECK-NEXT:   call void @__enzyme_memcpyadd_doubleda1sa1(double* %19, double* %"send_buf'", i64 %20)
; CHECK-NEXT:   tail call void @free(i8* nonnull %15)
; CHECK-NEXT:   br label %invertentry_post_post

; CHECK: invertentry_post_post:                            ; preds = %invertentry_post_root, %invertentry_post
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
