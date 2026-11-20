
FUNCTION_BEGIN __hexagon_divsf3
  {
    r2,p0 = sfrecipa(r0,r1)
    r4 = sffixupd(r0,r1)
    r3 = ##0x3f800000
  }
  {
    r5 = sffixupn(r0,r1)
    r3 -= sfmpy(r4,r2):lib
    r6 = ##0x80000000
    r7 = r3
  }
  {
    r2 += sfmpy(r3,r2):lib
    r3 = r7
    r6 = r5
    r0 = and(r6,r5)
  }
  {
    r3 -= sfmpy(r4,r2):lib
    r0 += sfmpy(r5,r2):lib
  }
  {
    r2 += sfmpy(r3,r2):lib
    r6 -= sfmpy(r0,r4):lib
  }
  {
    r0 += sfmpy(r6,r2):lib
  }
  {
    r5 -= sfmpy(r0,r4):lib
  }
  {
    r0 += sfmpy(r5,r2,p0):scale
    jumpr r31
  }
FUNCTION_END __hexagon_divsf3

.global __qdsp_divsf3 ; .set __qdsp_divsf3, __hexagon_divsf3
.global __hexagon_fast_divsf3 ; .set __hexagon_fast_divsf3, __hexagon_divsf3
.global __hexagon_fast2_divsf3 ; .set __hexagon_fast2_divsf3, __hexagon_divsf3
