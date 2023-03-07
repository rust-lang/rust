; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -S | FileCheck %s -check-prefix=MSG1
; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -S | FileCheck %s -check-prefix=MSG2
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify)" -enzyme-preopt=false -S | FileCheck %s -check-prefix=MSG1
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify)" -enzyme-preopt=false -S | FileCheck %s -check-prefix=MSG2

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
@.str = private unnamed_addr constant [33 x i8] c"Process %d: vald %f, valeurd %f\0A\00", align 1
@.str.1 = private unnamed_addr constant [31 x i8] c"Process %d: val %f, valeur %f\0A\00", align 1

; Function Attrs: nounwind uwtable
define dso_local void @msg1(float* %val1, float* %val2, i32 %numprocprec, i32 %numprocsuiv, i32 %etiquette) #0 {
entry:
  %r1 = alloca %struct.ompi_request_t*, align 8
  %s1 = alloca %struct.ompi_status_public_t, align 8
  %r2 = alloca %struct.ompi_request_t*, align 8
  %s2 = alloca %struct.ompi_status_public_t, align 8
  %0 = bitcast %struct.ompi_request_t** %r1 to i8*
  %1 = bitcast float* %val1 to i8*
  %call = call i32 @MPI_Isend(i8* %1, i32 1, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_real to %struct.ompi_datatype_t*), i32 %numprocsuiv, i32 %etiquette, %struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*), %struct.ompi_request_t** nonnull %r1) #4
  %2 = bitcast %struct.ompi_status_public_t* %s1 to i8*
  %call1 = call i32 @MPI_Wait(%struct.ompi_request_t** nonnull %r1, %struct.ompi_status_public_t* nonnull %s1) #4
  %3 = bitcast %struct.ompi_request_t** %r2 to i8*
  %4 = bitcast float* %val2 to i8*
  %call2 = call i32 @MPI_Irecv(i8* %4, i32 1, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_real to %struct.ompi_datatype_t*), i32 %numprocprec, i32 %etiquette, %struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*), %struct.ompi_request_t** nonnull %r2) #4
  %5 = bitcast %struct.ompi_status_public_t* %s2 to i8*
  %call3 = call i32 @MPI_Wait(%struct.ompi_request_t** nonnull %r2, %struct.ompi_status_public_t* nonnull %s2) #4
  ret void
}

declare dso_local i32 @MPI_Isend(i8*, i32, %struct.ompi_datatype_t*, i32, i32, %struct.ompi_communicator_t*, %struct.ompi_request_t**)

declare dso_local i32 @MPI_Wait(%struct.ompi_request_t**, %struct.ompi_status_public_t*) 

declare dso_local i32 @MPI_Irecv(i8*, i32, %struct.ompi_datatype_t*, i32, i32, %struct.ompi_communicator_t*, %struct.ompi_request_t**)

; Function Attrs: nounwind uwtable
define dso_local void @msg2(float* %val1, float* %val2, i32 %numprocprec, i32 %numprocsuiv, i32 %etiquette) #0 {
entry:
  %statut = alloca %struct.ompi_status_public_t, align 8
  %0 = bitcast %struct.ompi_status_public_t* %statut to i8*
  %1 = bitcast float* %val2 to i8*
  %call = call i32 @MPI_Recv(i8* %1, i32 1, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_real to %struct.ompi_datatype_t*), i32 %numprocprec, i32 %etiquette, %struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*), %struct.ompi_status_public_t* nonnull %statut) #4
  %2 = bitcast float* %val1 to i8*
  %call1 = call i32 @MPI_Send(i8* %2, i32 1, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_real to %struct.ompi_datatype_t*), i32 %numprocsuiv, i32 %etiquette, %struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*)) #4
  ret void
}

declare dso_local i32 @MPI_Recv(i8*, i32, %struct.ompi_datatype_t*, i32, i32, %struct.ompi_communicator_t*, %struct.ompi_status_public_t*)

declare dso_local i32 @MPI_Send(i8*, i32, %struct.ompi_datatype_t*, i32, i32, %struct.ompi_communicator_t*)

; Function Attrs: nounwind uwtable
define dso_local i32 @main(i32 %argc, i8** %argv) local_unnamed_addr #0 {
entry:
  %argc.addr = alloca i32, align 4
  %argv.addr = alloca i8**, align 8
  %nb_processus = alloca i32, align 4
  %rang = alloca i32, align 4
  %val = alloca float, align 4
  %valeur = alloca float, align 4
  %vald = alloca float, align 4
  %valeurd = alloca float, align 4
  store i32 %argc, i32* %argc.addr, align 4, !tbaa !2
  store i8** %argv, i8*** %argv.addr, align 8, !tbaa !6
  %0 = bitcast i32* %nb_processus to i8*
  %1 = bitcast i32* %rang to i8*
  %2 = bitcast float* %val to i8*
  %3 = bitcast float* %valeur to i8*
  %call = call i32 @MPI_Init(i32* nonnull %argc.addr, i8*** nonnull %argv.addr) #4
  %call1 = call i32 @MPI_Comm_rank(%struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*), i32* nonnull %rang) #4
  %call2 = call i32 @MPI_Comm_size(%struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*), i32* nonnull %nb_processus) #4
  %4 = load i32, i32* %nb_processus, align 4, !tbaa !2
  %5 = load i32, i32* %rang, align 4, !tbaa !2
  %add = add i32 %4, -1
  %sub = add i32 %add, %5
  %rem = srem i32 %sub, %4
  %add3 = add nsw i32 %5, 1
  %rem4 = srem i32 %add3, %4
  %add5 = add nsw i32 %5, 1000
  %conv = sitofp i32 %add5 to float
  store float %conv, float* %val, align 4, !tbaa !8
  %6 = bitcast float* %vald to i8*
  %7 = bitcast float* %valeurd to i8*
  %add6 = add nsw i32 %5, 2000
  %conv7 = sitofp i32 %add6 to float
  store float %conv7, float* %valeurd, align 4, !tbaa !8
  %cmp = icmp eq i32 %5, 0
  %.sink = select i1 %cmp, i8* bitcast (void (float*, float*, i32, i32, i32)* @msg1 to i8*), i8* bitcast (void (float*, float*, i32, i32, i32)* @msg2 to i8*)
  call void (i8*, ...) @__enzyme_fwddiff(i8* %.sink, float* nonnull %val, float* nonnull %vald, float* nonnull %valeur, float* nonnull %valeurd, i32 %rem, i32 %rem4, i32 100) #4
  %8 = load i32, i32* %rang, align 4, !tbaa !2
  %9 = load float, float* %vald, align 4, !tbaa !8
  %conv9 = fpext float %9 to double
  %10 = load float, float* %valeurd, align 4, !tbaa !8
  %conv10 = fpext float %10 to double
  %call11 = call i32 (i8*, ...) @printf(i8* nonnull dereferenceable(1) getelementptr inbounds ([33 x i8], [33 x i8]* @.str, i64 0, i64 0), i32 %8, double %conv9, double %conv10)
  %11 = load i32, i32* %rang, align 4, !tbaa !2
  %12 = load float, float* %val, align 4, !tbaa !8
  %conv12 = fpext float %12 to double
  %13 = load float, float* %valeur, align 4, !tbaa !8
  %conv13 = fpext float %13 to double
  %call14 = call i32 (i8*, ...) @printf(i8* nonnull dereferenceable(1) getelementptr inbounds ([31 x i8], [31 x i8]* @.str.1, i64 0, i64 0), i32 %11, double %conv12, double %conv13)
  %call15 = call i32 @MPI_Finalize() #4
  ret i32 0
}

declare dso_local i32 @MPI_Init(i32*, i8***)

declare dso_local i32 @MPI_Comm_rank(%struct.ompi_communicator_t*, i32*)

declare dso_local i32 @MPI_Comm_size(%struct.ompi_communicator_t*, i32*)

declare dso_local void @__enzyme_fwddiff(i8*, ...) 

; Function Attrs: nofree nounwind
declare dso_local i32 @printf(i8* nocapture readonly, ...) 

declare dso_local i32 @MPI_Finalize() 

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


; MSG1: define internal void @fwddiffemsg1(float* %val1, float* %"val1'", float* %val2, float* %"val2'", i32 %numprocprec, i32 %numprocsuiv, i32 %etiquette)
; MSG1-NEXT: entry:
; MSG1-NEXT:   %"r1'ipa" = alloca %struct.ompi_request_t*, align 8
; MSG1-NEXT:   store %struct.ompi_request_t* null, %struct.ompi_request_t** %"r1'ipa", align 8
; MSG1-NEXT:   %r1 = alloca %struct.ompi_request_t*, align 8
; MSG1-NEXT:   %s1 = alloca %struct.ompi_status_public_t, align 8
; MSG1-NEXT:   %"r2'ipa" = alloca %struct.ompi_request_t*, align 8
; MSG1-NEXT:   store %struct.ompi_request_t* null, %struct.ompi_request_t** %"r2'ipa", align 8
; MSG1-NEXT:   %r2 = alloca %struct.ompi_request_t*, align 8
; MSG1-NEXT:   %s2 = alloca %struct.ompi_status_public_t, align 8
; MSG1-NEXT:   %"'ipc" = bitcast float* %"val1'" to i8*
; MSG1-NEXT:   %0 = bitcast float* %val1 to i8*
; MSG1-NEXT:   %call = call i32 @MPI_Isend(i8* %0, i32 1, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_real to %struct.ompi_datatype_t*), i32 %numprocsuiv, i32 %etiquette, %struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*), %struct.ompi_request_t** nonnull %r1)
; MSG1-NEXT:   %1 = call i32 @MPI_Isend(i8* %"'ipc", i32 1, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_real to %struct.ompi_datatype_t*), i32 %numprocsuiv, i32 %etiquette, %struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*), %struct.ompi_request_t** %"r1'ipa")
; MSG1-NEXT:   %call1 = call i32 @MPI_Wait(%struct.ompi_request_t** nonnull %r1, %struct.ompi_status_public_t* nonnull %s1)
; MSG1-NEXT:   %2 = call i32 @MPI_Wait(%struct.ompi_request_t** %"r1'ipa", %struct.ompi_status_public_t* %s1)
; MSG1-NEXT:   %[[ipc3:.+]] = bitcast float* %"val2'" to i8*
; MSG1-NEXT:   %3 = bitcast float* %val2 to i8*
; MSG1-NEXT:   %call2 = call i32 @MPI_Irecv(i8* %3, i32 1, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_real to %struct.ompi_datatype_t*), i32 %numprocprec, i32 %etiquette, %struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*), %struct.ompi_request_t** nonnull %r2)
; MSG1-NEXT:   %4 = call i32 @MPI_Irecv(i8* %[[ipc3]], i32 1, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_real to %struct.ompi_datatype_t*), i32 %numprocprec, i32 %etiquette, %struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*), %struct.ompi_request_t** %"r2'ipa")
; MSG1-NEXT:   %call3 = call i32 @MPI_Wait(%struct.ompi_request_t** nonnull %r2, %struct.ompi_status_public_t* nonnull %s2)
; MSG1-NEXT:   %5 = call i32 @MPI_Wait(%struct.ompi_request_t** %"r2'ipa", %struct.ompi_status_public_t* %s2)
; MSG1-NEXT:   ret void
; MSG1-NEXT: }


; MSG2: define internal void @fwddiffemsg2(float* %val1, float* %"val1'", float* %val2, float* %"val2'", i32 %numprocprec, i32 %numprocsuiv, i32 %etiquette)
; MSG2-NEXT: entry:
; MSG2-NEXT:   %statut = alloca %struct.ompi_status_public_t, align 8
; MSG2-NEXT:   %0 = bitcast float* %val2 to i8*
; MSG2-NEXT:   %call = call i32 @MPI_Recv(i8* %0, i32 1, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_real to %struct.ompi_datatype_t*), i32 %numprocprec, i32 %etiquette, %struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*), %struct.ompi_status_public_t* nonnull %statut)
; MSG2-NEXT:   %[[ipc:.+]] = bitcast float* %"val1'" to i8*
; MSG2-NEXT:   %1 = bitcast float* %val1 to i8*
; MSG2-NEXT:   %call1 = call i32 @MPI_Send(i8* %1, i32 1, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_real to %struct.ompi_datatype_t*), i32 %numprocsuiv, i32 %etiquette, %struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*))
; MSG2-NEXT:   %2 = call i32 @MPI_Send(i8* %[[ipc]], i32 1, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_real to %struct.ompi_datatype_t*), i32 %numprocsuiv, i32 %etiquette, %struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*))
; MSG2-NEXT:   ret void
; MSG2-NEXT: }
