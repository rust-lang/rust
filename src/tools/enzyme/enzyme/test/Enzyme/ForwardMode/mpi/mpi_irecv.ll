; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -adce -S | FileCheck %s
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(adce)" -enzyme-preopt=false -S | FileCheck %s

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
  call void (i8*, ...) @__enzyme_fwddiff(i8* bitcast (void (i8**, i32, i32)* @meta to i8*), metadata !"enzyme_dup", i8** %ptr, i8** %dptr, i32 %numprocprec, i32 %etiquette)
  ret void
}

declare dso_local void @__enzyme_fwddiff(i8*, ...) 

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


; CHECK: define internal void @fwddiffemsg1(i8** %ptr, i8** %"ptr'", i32 %numprocprec, i32 %etiquette)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"r2'ipa" = alloca %struct.ompi_request_t*, align 8
; CHECK-NEXT:   store %struct.ompi_request_t* null, %struct.ompi_request_t** %"r2'ipa", align 8
; CHECK-NEXT:   %r2 = alloca %struct.ompi_request_t*, align 8
; CHECK-NEXT:   %s2 = alloca %struct.ompi_status_public_t, align 8
; CHECK-NEXT:   %"into'ipl" = load i8*, i8** %"ptr'"
; CHECK-NEXT:   %into = load i8*, i8** %ptr
; CHECK-NEXT:   %call2 = call i32 @MPI_Irecv(i8* %into, i32 1, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_real to %struct.ompi_datatype_t*), i32 %numprocprec, i32 %etiquette, %struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*), %struct.ompi_request_t** nonnull %r2)
; CHECK-NEXT:   %0 = call i32 @MPI_Irecv(i8* %"into'ipl", i32 1, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_real to %struct.ompi_datatype_t*), i32 %numprocprec, i32 %etiquette, %struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*), %struct.ompi_request_t** %"r2'ipa")
; CHECK-NEXT:   %call3 = call i32 @MPI_Wait(%struct.ompi_request_t** nonnull %r2, %struct.ompi_status_public_t* nonnull %s2)
; CHECK-NEXT:   %1 = call i32 @MPI_Wait(%struct.ompi_request_t** %"r2'ipa", %struct.ompi_status_public_t* %s2)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
