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

declare dso_local i32 @MPI_Isend(i8*, i32, %struct.ompi_datatype_t*, i32, i32, %struct.ompi_communicator_t*, %struct.ompi_request_t**) local_unnamed_addr #2

declare dso_local i32 @MPI_Wait(%struct.ompi_request_t**, %struct.ompi_status_public_t*) local_unnamed_addr #2

declare dso_local i32 @MPI_Irecv(i8*, i32, %struct.ompi_datatype_t*, i32, i32, %struct.ompi_communicator_t*, %struct.ompi_request_t**) local_unnamed_addr #2

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

declare dso_local i32 @MPI_Recv(i8*, i32, %struct.ompi_datatype_t*, i32, i32, %struct.ompi_communicator_t*, %struct.ompi_status_public_t*) local_unnamed_addr #2

declare dso_local i32 @MPI_Send(i8*, i32, %struct.ompi_datatype_t*, i32, i32, %struct.ompi_communicator_t*) local_unnamed_addr #2

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
  call void (i8*, ...) @__enzyme_autodiff(i8* %.sink, float* nonnull %val, float* nonnull %vald, float* nonnull %valeur, float* nonnull %valeurd, i32 %rem, i32 %rem4, i32 100) #4
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

declare dso_local i32 @MPI_Init(i32*, i8***) local_unnamed_addr #2

declare dso_local i32 @MPI_Comm_rank(%struct.ompi_communicator_t*, i32*) local_unnamed_addr #2

declare dso_local i32 @MPI_Comm_size(%struct.ompi_communicator_t*, i32*) local_unnamed_addr #2

declare dso_local void @__enzyme_autodiff(i8*, ...) local_unnamed_addr #2

; Function Attrs: nofree nounwind
declare dso_local i32 @printf(i8* nocapture readonly, ...) local_unnamed_addr #3

declare dso_local i32 @MPI_Finalize() local_unnamed_addr #2

attributes #0 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind willreturn }
attributes #2 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nofree nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
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

; CHECK: define internal void @diffemsg1(float* %val1, float* %"val1'", float* %val2, float* %"val2'", i32 %numprocprec, i32 %numprocsuiv, i32 %etiquette) 
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"r1'ipa" = alloca %struct.ompi_request_t*, align 8
; CHECK-NEXT:   store %struct.ompi_request_t* null, %struct.ompi_request_t** %"r1'ipa", align 8
; CHECK-NEXT:   %r1 = alloca %struct.ompi_request_t*, align 8
; CHECK-NEXT:   %s1 = alloca %struct.ompi_status_public_t, align 8
; CHECK-NEXT:   %"r2'ipa" = alloca %struct.ompi_request_t*, align 8
; CHECK-NEXT:   store %struct.ompi_request_t* null, %struct.ompi_request_t** %"r2'ipa", align 8
; CHECK-NEXT:   %r2 = alloca %struct.ompi_request_t*, align 8
; CHECK-NEXT:   %s2 = alloca %struct.ompi_status_public_t, align 8
; CHECK-NEXT:   %0 = bitcast float* %val1 to i8*
; CHECK-NEXT:   %malloccall3 = tail call i8* @malloc(i64 56)
; CHECK-NEXT:   %1 = bitcast i8* %malloccall3 to { i8*, i64, i8*, i64, i64, i8*, i8 }*
; CHECK-NEXT:   %2 = bitcast %struct.ompi_request_t** %"r1'ipa" to { i8*, i64, i8*, i64, i64, i8*, i8 }**
; CHECK-NEXT:   store { i8*, i64, i8*, i64, i64, i8*, i8 }* %1, { i8*, i64, i8*, i64, i64, i8*, i8 }** %2
; CHECK-NEXT:   %3 = alloca i32
; CHECK-NEXT:   %4 = call i32 @MPI_Type_size(i8* bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_real to i8*), i32* %3)
; CHECK-NEXT:   %5 = load i32, i32* %3
; CHECK-NEXT:   %6 = zext i32 %5 to i64
; CHECK-NEXT:   %malloccall4 = tail call i8* @malloc(i64 %6)
; CHECK-NEXT:   %7 = getelementptr inbounds { i8*, i64, i8*, i64, i64, i8*, i8 }, { i8*, i64, i8*, i64, i64, i8*, i8 }* %1, i64 0, i32 0
; CHECK-NEXT:   store i8* %malloccall4, i8** %7
; CHECK-NEXT:   %8 = getelementptr inbounds { i8*, i64, i8*, i64, i64, i8*, i8 }, { i8*, i64, i8*, i64, i64, i8*, i8 }* %1, i64 0, i32 1
; CHECK-NEXT:   store i64 1, i64* %8
; CHECK-NEXT:   %9 = getelementptr inbounds { i8*, i64, i8*, i64, i64, i8*, i8 }, { i8*, i64, i8*, i64, i64, i8*, i8 }* %1, i64 0, i32 2
; CHECK-NEXT:   store i8* bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_real to i8*), i8** %9
; CHECK-NEXT:   %10 = getelementptr inbounds { i8*, i64, i8*, i64, i64, i8*, i8 }, { i8*, i64, i8*, i64, i64, i8*, i8 }* %1, i64 0, i32 3
; CHECK-NEXT:   %11 = zext i32 %numprocsuiv to i64
; CHECK-NEXT:   store i64 %11, i64* %10
; CHECK-NEXT:   %12 = getelementptr inbounds { i8*, i64, i8*, i64, i64, i8*, i8 }, { i8*, i64, i8*, i64, i64, i8*, i8 }* %1, i64 0, i32 4
; CHECK-NEXT:   %13 = zext i32 %etiquette to i64
; CHECK-NEXT:   store i64 %13, i64* %12
; CHECK-NEXT:   %14 = getelementptr inbounds { i8*, i64, i8*, i64, i64, i8*, i8 }, { i8*, i64, i8*, i64, i64, i8*, i8 }* %1, i64 0, i32 5
; CHECK-NEXT:   store i8* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to i8*), i8** %14
; CHECK-NEXT:   %15 = getelementptr inbounds { i8*, i64, i8*, i64, i64, i8*, i8 }, { i8*, i64, i8*, i64, i64, i8*, i8 }* %1, i64 0, i32 6
; CHECK-NEXT:   store i8 1, i8* %15
; CHECK-NEXT:   %call = call i32 @MPI_Isend(i8* %0, i32 1, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_real to %struct.ompi_datatype_t*), i32 %numprocsuiv, i32 %etiquette, %struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*), %struct.ompi_request_t** nonnull %r1)
; CHECK-NEXT:   %call1 = call i32 @MPI_Wait(%struct.ompi_request_t** nonnull %r1, %struct.ompi_status_public_t* nonnull %s1)
; CHECK-NEXT:   %"'ipc" = bitcast float* %"val2'" to i8*
; CHECK-NEXT:   %16 = bitcast float* %val2 to i8*
; CHECK-NEXT:   %malloccall = tail call i8* @malloc(i64 56)
; CHECK-NEXT:   %17 = bitcast i8* %malloccall to { i8*, i64, i8*, i64, i64, i8*, i8 }*
; CHECK-NEXT:   %18 = bitcast %struct.ompi_request_t** %"r2'ipa" to { i8*, i64, i8*, i64, i64, i8*, i8 }**
; CHECK-NEXT:   store { i8*, i64, i8*, i64, i64, i8*, i8 }* %17, { i8*, i64, i8*, i64, i64, i8*, i8 }** %18
; CHECK-NEXT:   %19 = getelementptr inbounds { i8*, i64, i8*, i64, i64, i8*, i8 }, { i8*, i64, i8*, i64, i64, i8*, i8 }* %17, i64 0, i32 0
; CHECK-NEXT:   store i8* %"'ipc", i8** %19
; CHECK-NEXT:   %20 = getelementptr inbounds { i8*, i64, i8*, i64, i64, i8*, i8 }, { i8*, i64, i8*, i64, i64, i8*, i8 }* %17, i64 0, i32 1
; CHECK-NEXT:   store i64 1, i64* %20
; CHECK-NEXT:   %21 = getelementptr inbounds { i8*, i64, i8*, i64, i64, i8*, i8 }, { i8*, i64, i8*, i64, i64, i8*, i8 }* %17, i64 0, i32 2
; CHECK-NEXT:   store i8* bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_real to i8*), i8** %21
; CHECK-NEXT:   %22 = getelementptr inbounds { i8*, i64, i8*, i64, i64, i8*, i8 }, { i8*, i64, i8*, i64, i64, i8*, i8 }* %17, i64 0, i32 3
; CHECK-NEXT:   %23 = zext i32 %numprocprec to i64
; CHECK-NEXT:   store i64 %23, i64* %22
; CHECK-NEXT:   %24 = getelementptr inbounds { i8*, i64, i8*, i64, i64, i8*, i8 }, { i8*, i64, i8*, i64, i64, i8*, i8 }* %17, i64 0, i32 4
; CHECK-NEXT:   %25 = zext i32 %etiquette to i64
; CHECK-NEXT:   store i64 %25, i64* %24
; CHECK-NEXT:   %26 = getelementptr inbounds { i8*, i64, i8*, i64, i64, i8*, i8 }, { i8*, i64, i8*, i64, i64, i8*, i8 }* %17, i64 0, i32 5
; CHECK-NEXT:   store i8* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to i8*), i8** %26
; CHECK-NEXT:   %27 = getelementptr inbounds { i8*, i64, i8*, i64, i64, i8*, i8 }, { i8*, i64, i8*, i64, i64, i8*, i8 }* %17, i64 0, i32 6
; CHECK-NEXT:   store i8 2, i8* %27
; CHECK-NEXT:   %call2 = call i32 @MPI_Irecv(i8* %16, i32 1, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_real to %struct.ompi_datatype_t*), i32 %numprocprec, i32 %etiquette, %struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*), %struct.ompi_request_t** nonnull %r2)
; CHECK-NEXT:   %call3 = call i32 @MPI_Wait(%struct.ompi_request_t** nonnull %r2, %struct.ompi_status_public_t* nonnull %s2) #5
; CHECK-NEXT:   %28 = bitcast %struct.ompi_request_t** %"r2'ipa" to { i8*, i64, i8*, i64, i64, i8*, i8 }**
; CHECK-NEXT:   %29 = load { i8*, i64, i8*, i64, i64, i8*, i8 }*, { i8*, i64, i8*, i64, i64, i8*, i8 }** %28
; CHECK-NEXT:   %30 = load { i8*, i64, i8*, i64, i64, i8*, i8 }, { i8*, i64, i8*, i64, i64, i8*, i8 }* %29
; CHECK-NEXT:   %31 = bitcast { i8*, i64, i8*, i64, i64, i8*, i8 }* %29 to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %31)
; CHECK-NEXT:   %32 = extractvalue { i8*, i64, i8*, i64, i64, i8*, i8 } %30, 0
; CHECK-NEXT:   %33 = extractvalue { i8*, i64, i8*, i64, i64, i8*, i8 } %30, 1
; CHECK-NEXT:   %34 = extractvalue { i8*, i64, i8*, i64, i64, i8*, i8 } %30, 2
; CHECK-NEXT:   %35 = extractvalue { i8*, i64, i8*, i64, i64, i8*, i8 } %30, 3
; CHECK-NEXT:   %36 = extractvalue { i8*, i64, i8*, i64, i64, i8*, i8 } %30, 4
; CHECK-NEXT:   %37 = extractvalue { i8*, i64, i8*, i64, i64, i8*, i8 } %30, 5
; CHECK-NEXT:   %38 = extractvalue { i8*, i64, i8*, i64, i64, i8*, i8 } %30, 6
; CHECK-NEXT:   call void @__enzyme_differential_mpi_wait(i8* %32, i64 %33, i8* %34, i64 %35, i64 %36, i8* %37, i8 %38, %struct.ompi_request_t** %"r2'ipa")
; CHECK-NEXT:   %39 = alloca %struct.ompi_status_public_t
; CHECK-NEXT:   %40 = call i32 @MPI_Wait(%struct.ompi_request_t** %"r2'ipa", %struct.ompi_status_public_t* %39)
; CHECK-NEXT:   %41 = alloca i32
; CHECK-NEXT:   %42 = call i32 @MPI_Type_size(i8* bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_real to i8*), i32* %41)
; CHECK-NEXT:   %43 = load i32, i32* %41
; CHECK-NEXT:   %44 = zext i32 %43 to i64
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* %"'ipc", i8 0, i64 %44, i1 false)
; CHECK-NEXT:   %45 = bitcast %struct.ompi_request_t** %"r1'ipa" to { i8*, i64, i8*, i64, i64, i8*, i8 }**
; CHECK-NEXT:   %46 = load { i8*, i64, i8*, i64, i64, i8*, i8 }*, { i8*, i64, i8*, i64, i64, i8*, i8 }** %45
; CHECK-NEXT:   %47 = load { i8*, i64, i8*, i64, i64, i8*, i8 }, { i8*, i64, i8*, i64, i64, i8*, i8 }* %46
; CHECK-NEXT:   %48 = bitcast { i8*, i64, i8*, i64, i64, i8*, i8 }* %46 to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %48)
; CHECK-NEXT:   %49 = extractvalue { i8*, i64, i8*, i64, i64, i8*, i8 } %47, 0
; CHECK-NEXT:   %50 = extractvalue { i8*, i64, i8*, i64, i64, i8*, i8 } %47, 1
; CHECK-NEXT:   %51 = extractvalue { i8*, i64, i8*, i64, i64, i8*, i8 } %47, 2
; CHECK-NEXT:   %52 = extractvalue { i8*, i64, i8*, i64, i64, i8*, i8 } %47, 3
; CHECK-NEXT:   %53 = extractvalue { i8*, i64, i8*, i64, i64, i8*, i8 } %47, 4
; CHECK-NEXT:   %54 = extractvalue { i8*, i64, i8*, i64, i64, i8*, i8 } %47, 5
; CHECK-NEXT:   %55 = extractvalue { i8*, i64, i8*, i64, i64, i8*, i8 } %47, 6
; CHECK-NEXT:   call void @__enzyme_differential_mpi_wait(i8* %49, i64 %50, i8* %51, i64 %52, i64 %53, i8* %54, i8 %55, %struct.ompi_request_t** %"r1'ipa")
; CHECK-NEXT:   %56 = alloca %struct.ompi_status_public_t
; CHECK-NEXT:   %57 = call i32 @MPI_Wait(%struct.ompi_request_t** %"r1'ipa", %struct.ompi_status_public_t* %56)
; CHECK-NEXT:   %58 = alloca i32
; CHECK-NEXT:   %59 = call i32 @MPI_Type_size(i8* bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_real to i8*), i32* %58)
; CHECK-NEXT:   %60 = bitcast i8* %malloccall4 to float*
; CHECK-NEXT:   %_unwrap = load i32, i32* %58
; CHECK-NEXT:   %_unwrap6 = zext i32 %_unwrap to i64
; CHECK-NEXT:   %61 = udiv i64 %_unwrap6, 4
; CHECK-NEXT:   call void @__enzyme_memcpyadd_floatda1sa1(float* %60, float* %"val1'", i64 %61)
; CHECK-NEXT:   tail call void @free(i8* nonnull %malloccall4)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define void @__enzyme_differential_mpi_wait(i8* %buf, i64 %count, i8* %datatype, i64 %source, i64 %tag, i8* %comm, i8 %fn, %struct.ompi_request_t** %d_req) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = trunc i64 %count to i32
; CHECK-NEXT:   %1 = bitcast i8* %datatype to %struct.ompi_datatype_t*
; CHECK-NEXT:   %2 = trunc i64 %source to i32
; CHECK-NEXT:   %3 = trunc i64 %tag to i32
; CHECK-NEXT:   %4 = bitcast i8* %comm to %struct.ompi_communicator_t*
; CHECK-NEXT:   %5 = icmp eq i8 %fn, 1
; CHECK-NEXT:   br i1 %5, label %invertISend, label %invertIRecv

; CHECK: invertISend:                                      ; preds = %entry
; CHECK-NEXT:   %6 = call i32 @MPI_Irecv(i8* %buf, i32 %0, %struct.ompi_datatype_t* %1, i32 %2, i32 %3, %struct.ompi_communicator_t* %4, %struct.ompi_request_t** %d_req)
; CHECK-NEXT:   ret void

; CHECK: invertIRecv:                                      ; preds = %entry
; CHECK-NEXT:   %7 = call i32 @MPI_Isend(i8* %buf, i32 %0, %struct.ompi_datatype_t* %1, i32 %2, i32 %3, %struct.ompi_communicator_t* %4, %struct.ompi_request_t** %d_req)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }