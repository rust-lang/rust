; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -S | FileCheck %s

@_ZTId = external dso_local constant i8*

define i8* @_ZNK4implIdE4typeEv()  {
  ret i8* bitcast (i8** @_ZTId to i8*)
}

declare i8* @_Z17__enzyme_virtualreverse(i8*)

define void @_Z18wrapper_1body_intsv()  {
  %a = call i8* @_Z17__enzyme_virtualreverse(i8* bitcast (i8* ()* @_ZNK4implIdE4typeEv to i8*))
  ret void
}

; CHECK: define internal { i8*, i8*, i8* } @augmented__ZNK4implIdE4typeEv()
; CHECK-NEXT:   %1 = alloca { i8*, i8*, i8* }
; CHECK-NEXT:   %2 = getelementptr inbounds { i8*, i8*, i8* }, { i8*, i8*, i8* }* %1, i32 0, i32 0
; CHECK-NEXT:   store i8* null, i8** %2
; CHECK-NEXT:   %3 = getelementptr inbounds { i8*, i8*, i8* }, { i8*, i8*, i8* }* %1, i32 0, i32 1
; CHECK-NEXT:   store i8* bitcast (i8** @_ZTId to i8*), i8** %3
; CHECK-NEXT:   %4 = getelementptr inbounds { i8*, i8*, i8* }, { i8*, i8*, i8* }* %1, i32 0, i32 2
; CHECK-NEXT:   store i8* bitcast (i8** @_ZTId_shadow to i8*), i8** %4
; CHECK-NEXT:   %5 = load { i8*, i8*, i8* }, { i8*, i8*, i8* }* %1
; CHECK-NEXT:   ret { i8*, i8*, i8* } %5
; CHECK-NEXT: }
