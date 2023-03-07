; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

; ModuleID = 'test/mpi2.c'
source_filename = "test/mpi2.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.ompi_predefined_datatype_t = type opaque
%struct.ompi_predefined_communicator_t = type opaque
%struct.ompi_request_t = type opaque
%struct.ompi_status_public_t = type { i32, i32, i32, i32, i64 }
%struct.ompi_datatype_t = type opaque
%struct.ompi_communicator_t = type opaque

@ompi_mpi_real = external dso_local global %struct.ompi_predefined_datatype_t, align 1
@ompi_mpi_comm_world = external dso_local global %struct.ompi_predefined_communicator_t, align 1

define void @msg1(i8** %ptr, i32 %numprocprec, i32 %etiquette) {
entry:
  %myRank.i.i.i = alloca i32
  %r2 = alloca %struct.ompi_request_t*, align 8
  %s2 = alloca %struct.ompi_status_public_t, align 8
  %into = load i8*, i8** %ptr
  %call.i.i.i = call i32 @MPI_Comm_rank(%struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*), i32* nonnull %myRank.i.i.i)
  %call2 = call i32 @MPI_Irecv(i8* %into, i32 1, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_real to %struct.ompi_datatype_t*), i32 %numprocprec, i32 %etiquette, %struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*), %struct.ompi_request_t** nonnull %r2) #4
  %call3 = call i32 @MPI_Wait(%struct.ompi_request_t** nonnull %r2, %struct.ompi_status_public_t* nonnull %s2) #4
  ret void
}

define void @meta(i8** %ptr, i32 %numprocprec, i32 %etiquette) {
entry:
  call void @msg1(i8** %ptr, i32 %numprocprec, i32 %etiquette)
  store i8* null, i8** %ptr
  ret void
}

declare dso_local i32 @MPI_Comm_rank(%struct.ompi_communicator_t*, i32*)

declare dso_local i32 @MPI_Isend(i8*, i32, %struct.ompi_datatype_t*, i32, i32, %struct.ompi_communicator_t*, %struct.ompi_request_t**)

declare dso_local i32 @MPI_Wait(%struct.ompi_request_t**, %struct.ompi_status_public_t*) 

declare dso_local i32 @MPI_Irecv(i8*, i32, %struct.ompi_datatype_t*, i32, i32, %struct.ompi_communicator_t*, %struct.ompi_request_t**)

define void @caller(i8** %ptr, i8** %dptr, i32 %numprocprec, i32 %etiquette) {
entry:  
  call void (i8*, ...) @__enzyme_autodiff(i8* bitcast (void (i8**, i32, i32)* @meta to i8*), metadata !"enzyme_dup", i8** %ptr, i8** %dptr, i32 %numprocprec, i32 %etiquette)
  ret void
}

declare dso_local void @__enzyme_autodiff(i8*, ...) 

attributes #0 = { nounwind uwtable }
attributes #4 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 10.0.1 (git@github.com:llvm/llvm-project ef32c611aa214dea855364efd7ba451ec5ec3f74)"}
!2 = !{!3, !3, i64 0}
!3 = !{!"int", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
!6 = !{!7, !7, i64 0}
!7 = !{!"any pointer", !4, i64 0}
!8 = !{!9, !9, i64 0}
!9 = !{!"float", !4, i64 0}

; CHECK: define internal void @diffemsg1(i8** %ptr, i8** %"ptr'", i32 %numprocprec, i32 %etiquette, { { i8*, i64, i8*, i64, i64, i8*, i8, i8* }*, i8*, i8* } %tapeArg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = alloca i32
; CHECK-NEXT:   %1 = alloca %struct.ompi_status_public_t
; CHECK-NEXT:   %malloccall = extractvalue { { i8*, i64, i8*, i64, i64, i8*, i8, i8* }*, i8*, i8* } %tapeArg, 2
; CHECK-NEXT:   %"malloccall'mi" = extractvalue { { i8*, i64, i8*, i64, i64, i8*, i8, i8* }*, i8*, i8* } %tapeArg, 1
; CHECK-NEXT:   %"r2'ipc" = bitcast i8* %"malloccall'mi" to %struct.ompi_request_t**
; CHECK-NEXT:   %r2 = bitcast i8* %malloccall to %struct.ompi_request_t**
; CHECK-NEXT:   %2 = extractvalue { { i8*, i64, i8*, i64, i64, i8*, i8, i8* }*, i8*, i8* } %tapeArg, 0
; CHECK-NEXT:   %3 = icmp eq { i8*, i64, i8*, i64, i64, i8*, i8, i8* }* %2, null
; CHECK-NEXT:   br i1 %3, label %invertentry_end, label %invertentry_nonnull

; CHECK: invertentry_nonnull:                              ; preds = %entry
; CHECK-NEXT:   %4 = load { i8*, i64, i8*, i64, i64, i8*, i8, i8* }, { i8*, i64, i8*, i64, i64, i8*, i8, i8* }* %2
; CHECK-NEXT:   %5 = extractvalue { i8*, i64, i8*, i64, i64, i8*, i8, i8* } %4, 0
; CHECK-NEXT:   %6 = extractvalue { i8*, i64, i8*, i64, i64, i8*, i8, i8* } %4, 1
; CHECK-NEXT:   %7 = extractvalue { i8*, i64, i8*, i64, i64, i8*, i8, i8* } %4, 2
; CHECK-NEXT:   %8 = extractvalue { i8*, i64, i8*, i64, i64, i8*, i8, i8* } %4, 3
; CHECK-NEXT:   %9 = extractvalue { i8*, i64, i8*, i64, i64, i8*, i8, i8* } %4, 4
; CHECK-NEXT:   %10 = extractvalue { i8*, i64, i8*, i64, i64, i8*, i8, i8* } %4, 5
; CHECK-NEXT:   %11 = extractvalue { i8*, i64, i8*, i64, i64, i8*, i8, i8* } %4, 6
; CHECK-NEXT:   %12 = trunc i64 %6 to i32
; CHECK-NEXT:   %13 = bitcast i8* %7 to %struct.ompi_datatype_t*
; CHECK-NEXT:   %14 = trunc i64 %8 to i32
; CHECK-NEXT:   %15 = trunc i64 %9 to i32
; CHECK-NEXT:   %16 = bitcast i8* %10 to %struct.ompi_communicator_t*
; CHECK-NEXT:   %17 = icmp eq i8 %11, 1
; CHECK-NEXT:   br i1 %17, label %invertISend.i, label %invertIRecv.i

; CHECK: invertISend.i:                                    ; preds = %invertentry_nonnull
; CHECK-NEXT:   %18 = call i32 @MPI_Irecv(i8* %5, i32 %12, %struct.ompi_datatype_t* %13, i32 %14, i32 %15, %struct.ompi_communicator_t* %16, %struct.ompi_request_t** %r2)
; CHECK-NEXT:   br label %invertentry_end

; CHECK: invertIRecv.i:                                    ; preds = %invertentry_nonnull
; CHECK-NEXT:   %19 = call i32 @MPI_Isend(i8* %5, i32 %12, %struct.ompi_datatype_t* %13, i32 %14, i32 %15, %struct.ompi_communicator_t* %16, %struct.ompi_request_t** %r2)
; CHECK-NEXT:   br label %invertentry_end

; CHECK: invertentry_end:                                  ; preds = %invertISend.i, %invertIRecv.i, %entry
; CHECK-NEXT:   %20 = bitcast %struct.ompi_request_t** %"r2'ipc" to { i8*, i64, i8*, i64, i64, i8*, i8, i8* }**
; CHECK-NEXT:   %21 = load { i8*, i64, i8*, i64, i64, i8*, i8, i8* }*, { i8*, i64, i8*, i64, i64, i8*, i8, i8* }** %20
; CHECK-NEXT:   %22 = getelementptr inbounds { i8*, i64, i8*, i64, i64, i8*, i8, i8* }, { i8*, i64, i8*, i64, i64, i8*, i8, i8* }* %21, i64 0, i32 0
; CHECK-NEXT:   %23 = load i8*, i8** %22
; CHECK-NEXT:   %24 = getelementptr inbounds { i8*, i64, i8*, i64, i64, i8*, i8, i8* }, { i8*, i64, i8*, i64, i64, i8*, i8, i8* }* %21, i64 0, i32 7
; CHECK-NEXT:   %25 = load i8*, i8** %24
; CHECK-NEXT:   %26 = bitcast %struct.ompi_request_t** %"r2'ipc" to i8**
; CHECK-NEXT:   store i8* %25, i8** %26
; CHECK-NEXT:   %27 = call i32 @MPI_Type_size(i8* bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_real to i8*), i32* %0)
; CHECK-NEXT:   %28 = load i32, i32* %0
; CHECK-NEXT:   %29 = call i32 @MPI_Wait(%struct.ompi_request_t** %r2, %struct.ompi_status_public_t* %1)
; CHECK-NEXT:   %30 = zext i32 %28 to i64
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* nonnull %23, i8 0, i64 %30, i1 false)
; CHECK-NEXT:   %31 = bitcast { i8*, i64, i8*, i64, i64, i8*, i8, i8* }* %21 to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %31)
; CHECK-NEXT:   tail call void @free(i8* nonnull %"malloccall'mi")
; CHECK-NEXT:   tail call void @free(i8* %malloccall)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }