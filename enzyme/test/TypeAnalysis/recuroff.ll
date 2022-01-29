; RUN: %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=r -o /dev/null | FileCheck %s


@ptr = private unnamed_addr global [5000 x i64] zeroinitializer, align 1

define void @callee(i64* %x, i64 %off) {
entry:
  %gep = getelementptr inbounds i64, i64* %x, i64 %off
  %ld = load i64, i64* %gep, align 8, !tbaa !8
  %add = add i64 %off, 1
  call void @callee(i64* %x, i64 %add)
  ret void
}

define void @r(i64* %x) {
entry:
  call void @callee(i64* %x, i64 23)
  ret void
}

!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C++ TBAA"}
!7 = !{!"double", !5, i64 0}
!8 = !{!7, !7, i64 0}


; CHECK: callee - {} |{[-1]:Pointer}:{} {[-1]:Integer}:{23,} 
; CHECK-NEXT: i64* %x: {[-1]:Pointer, [-1,184]:Float@double, [-1,192]:Float@double, [-1,200]:Float@double, [-1,208]:Float@double, [-1,216]:Float@double, [-1,224]:Float@double, [-1,232]:Float@double, [-1,240]:Float@double, [-1,248]:Float@double, [-1,256]:Float@double, [-1,264]:Float@double, [-1,272]:Float@double, [-1,280]:Float@double, [-1,288]:Float@double, [-1,296]:Float@double, [-1,304]:Float@double, [-1,312]:Float@double, [-1,320]:Float@double, [-1,328]:Float@double, [-1,336]:Float@double, [-1,344]:Float@double, [-1,352]:Float@double, [-1,360]:Float@double, [-1,368]:Float@double, [-1,376]:Float@double, [-1,384]:Float@double, [-1,392]:Float@double, [-1,400]:Float@double, [-1,408]:Float@double, [-1,416]:Float@double, [-1,424]:Float@double, [-1,432]:Float@double, [-1,440]:Float@double, [-1,448]:Float@double, [-1,456]:Float@double, [-1,464]:Float@double, [-1,472]:Float@double, [-1,480]:Float@double, [-1,488]:Float@double, [-1,496]:Float@double}
; CHECK-NEXT: i64 %off: {[-1]:Integer}
; CHECK-NEXT: entry
; CHECK-NEXT:   %gep = getelementptr inbounds i64, i64* %x, i64 %off: {[-1]:Pointer, [-1,0]:Float@double, [-1,8]:Float@double, [-1,16]:Float@double, [-1,24]:Float@double, [-1,32]:Float@double, [-1,40]:Float@double, [-1,48]:Float@double, [-1,56]:Float@double, [-1,64]:Float@double, [-1,72]:Float@double, [-1,80]:Float@double, [-1,88]:Float@double, [-1,96]:Float@double, [-1,104]:Float@double, [-1,112]:Float@double, [-1,120]:Float@double, [-1,128]:Float@double, [-1,136]:Float@double, [-1,144]:Float@double, [-1,152]:Float@double, [-1,160]:Float@double, [-1,168]:Float@double, [-1,176]:Float@double, [-1,184]:Float@double, [-1,192]:Float@double, [-1,200]:Float@double, [-1,208]:Float@double, [-1,216]:Float@double, [-1,224]:Float@double, [-1,232]:Float@double, [-1,240]:Float@double, [-1,248]:Float@double, [-1,256]:Float@double, [-1,264]:Float@double, [-1,272]:Float@double, [-1,280]:Float@double, [-1,288]:Float@double, [-1,296]:Float@double, [-1,304]:Float@double, [-1,312]:Float@double}
; CHECK-NEXT:   %ld = load i64, i64* %gep, align 8, !tbaa !0: {[-1]:Float@double}
; CHECK-NEXT:   %add = add i64 %off, 1: {[-1]:Integer}
; CHECK-NEXT:   call void @callee(i64* %x, i64 %add): {}
; CHECK-NEXT:   ret void: {}
; CHECK-NEXT: callee - {} |{[-1]:Pointer, [-1,184]:Float@double}:{} {[-1]:Integer}:{24,} 
; CHECK-NEXT: i64* %x: {[-1]:Pointer, [-1,184]:Float@double, [-1,192]:Float@double, [-1,200]:Float@double, [-1,208]:Float@double, [-1,216]:Float@double, [-1,224]:Float@double, [-1,232]:Float@double, [-1,240]:Float@double, [-1,248]:Float@double, [-1,256]:Float@double, [-1,264]:Float@double, [-1,272]:Float@double, [-1,280]:Float@double, [-1,288]:Float@double, [-1,296]:Float@double, [-1,304]:Float@double, [-1,312]:Float@double, [-1,320]:Float@double, [-1,328]:Float@double, [-1,336]:Float@double, [-1,344]:Float@double, [-1,352]:Float@double, [-1,360]:Float@double, [-1,368]:Float@double, [-1,376]:Float@double, [-1,384]:Float@double, [-1,392]:Float@double, [-1,400]:Float@double, [-1,408]:Float@double, [-1,416]:Float@double, [-1,424]:Float@double, [-1,432]:Float@double, [-1,440]:Float@double, [-1,448]:Float@double, [-1,456]:Float@double, [-1,464]:Float@double, [-1,472]:Float@double, [-1,480]:Float@double, [-1,488]:Float@double, [-1,496]:Float@double}
; CHECK-NEXT: i64 %off: {[-1]:Integer}
; CHECK-NEXT: entry
; CHECK-NEXT:   %gep = getelementptr inbounds i64, i64* %x, i64 %off: {[-1]:Pointer, [-1,0]:Float@double, [-1,8]:Float@double, [-1,16]:Float@double, [-1,24]:Float@double, [-1,32]:Float@double, [-1,40]:Float@double, [-1,48]:Float@double, [-1,56]:Float@double, [-1,64]:Float@double, [-1,72]:Float@double, [-1,80]:Float@double, [-1,88]:Float@double, [-1,96]:Float@double, [-1,104]:Float@double, [-1,112]:Float@double, [-1,120]:Float@double, [-1,128]:Float@double, [-1,136]:Float@double, [-1,144]:Float@double, [-1,152]:Float@double, [-1,160]:Float@double, [-1,168]:Float@double, [-1,176]:Float@double, [-1,184]:Float@double, [-1,192]:Float@double, [-1,200]:Float@double, [-1,208]:Float@double, [-1,216]:Float@double, [-1,224]:Float@double, [-1,232]:Float@double, [-1,240]:Float@double, [-1,248]:Float@double, [-1,256]:Float@double, [-1,264]:Float@double, [-1,272]:Float@double, [-1,280]:Float@double, [-1,288]:Float@double, [-1,296]:Float@double, [-1,304]:Float@double}
; CHECK-NEXT:   %ld = load i64, i64* %gep, align 8, !tbaa !0: {[-1]:Float@double}
; CHECK-NEXT:   %add = add i64 %off, 1: {[-1]:Integer}
; CHECK-NEXT:   call void @callee(i64* %x, i64 %add): {}
; CHECK-NEXT:   ret void: {}
