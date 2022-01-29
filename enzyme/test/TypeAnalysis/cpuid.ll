; RUN: %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=caller -o /dev/null | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @caller(i32* %l1) {
entry:
  %inp = load i32, i32* %l1, align 4
  %cu = call { i32, i32, i32, i32 } asm sideeffect "cpuid", "={ax},={bx},={cx},={dx},0,2,~{dirflag},~{fpsr},~{flags}"(i32 4, i32 %inp)
  %asmresult = extractvalue { i32, i32, i32, i32 } %cu, 0
  %asmresult8 = extractvalue { i32, i32, i32, i32 } %cu, 1
  %asmresult9 = extractvalue { i32, i32, i32, i32 } %cu, 2
  %asmresult10 = extractvalue { i32, i32, i32, i32 } %cu, 3
  ret void
}

; CHECK: caller - {} |{[-1]:Pointer}:{} 
; CHECK-NEXT: i32* %l1: {[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer, [-1,2]:Integer, [-1,3]:Integer}
; CHECK-NEXT: entry
; CHECK-NEXT:   %inp = load i32, i32* %l1, align 4: {[-1]:Integer}
; CHECK-NEXT:   %cu = call { i32, i32, i32, i32 } asm sideeffect "cpuid", "={ax},={bx},={cx},={dx},0,2,~{dirflag},~{fpsr},~{flags}"(i32 4, i32 %inp): {[-1]:Integer}
; CHECK-NEXT:   %asmresult = extractvalue { i32, i32, i32, i32 } %cu, 0: {[-1]:Integer}
; CHECK-NEXT:   %asmresult8 = extractvalue { i32, i32, i32, i32 } %cu, 1: {[-1]:Integer}
; CHECK-NEXT:   %asmresult9 = extractvalue { i32, i32, i32, i32 } %cu, 2: {[-1]:Integer}
; CHECK-NEXT:   %asmresult10 = extractvalue { i32, i32, i32, i32 } %cu, 3: {[-1]:Integer}
; CHECK-NEXT:   ret void: {}
