; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -adce -simplifycfg -S | FileCheck %s

%DynamicsStruct = type { i8**, void (%DynamicsStruct*)** }

@someGlobal = internal constant i8* bitcast (void (%DynamicsStruct*)* @asdf to i8*)

define internal void @asdf(%DynamicsStruct* %arg)  {
bb:
  %i = getelementptr inbounds %DynamicsStruct, %DynamicsStruct* %arg, i64 0, i32 0
  store i8** @someGlobal, i8*** %i, align 8
  %i5 = getelementptr inbounds %DynamicsStruct, %DynamicsStruct* %arg, i64 0, i32 1
  %i6 = load void (%DynamicsStruct*)**, void (%DynamicsStruct*)*** %i5, align 8
  %i8 = load void (%DynamicsStruct*)*, void (%DynamicsStruct*)** %i6, align 8
  tail call void %i8(%DynamicsStruct* %arg)
  ret void
}

declare i8* @_Z17__enzyme_virtualreversePv(...)

define internal void @_Z19testSensitivitiesADv() {
bb40:
  call i8* (...) @_Z17__enzyme_virtualreversePv(void (%DynamicsStruct*)* @asdf)
  ret void
}

; CHECK: define internal i8* @augmented_asdf(%DynamicsStruct* %arg, %DynamicsStruct* %"arg'")

; CHECK: define internal void @diffeasdf.1(%DynamicsStruct* %arg, %DynamicsStruct* %"arg'", i8* %tapeArg)
