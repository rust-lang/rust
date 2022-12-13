; RUN: if [ %llvmver -ge 10 ]; then %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=caller -o /dev/null | FileCheck %s; fi

declare double @f()

define void @caller() {
entry:
  %c = call double @f()
  %n = fneg double %c
  ret void
}

; CHECK: caller - {} |
; CHECK-NEXT: entry
; CHECK-NEXT:   %c = call double @f(): {[-1]:Float@double}
; CHECK-NEXT:   %n = fneg double %c: {[-1]:Float@double}
; CHECK-NEXT:   ret void: {}
