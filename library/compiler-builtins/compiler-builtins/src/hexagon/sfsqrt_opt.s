FUNCTION_BEGIN __hexagon_sqrtf
  {
    r3,p0 = sfinvsqrta(r0)
    r5 = sffixupr(r0)
    r4 = ##0x3f000000
    r1:0 = combine(#0,#0)
  }
  {
    r0 += sfmpy(r3,r5):lib
    r1 += sfmpy(r3,r4):lib
    r2 = r4
    r3 = r5
  }
  {
    r2 -= sfmpy(r0,r1):lib
    p1 = sfclass(r5,#1)

  }
  {
    r0 += sfmpy(r0,r2):lib
    r1 += sfmpy(r1,r2):lib
    r2 = r4
    r3 = r5
  }
  {
    r2 -= sfmpy(r0,r1):lib
    r3 -= sfmpy(r0,r0):lib
  }
  {
    r0 += sfmpy(r1,r3):lib
    r1 += sfmpy(r1,r2):lib
    r2 = r4
    r3 = r5
  }
  {

    r3 -= sfmpy(r0,r0):lib
    if (p1) r0 = or(r0,r5)
  }
  {
    r0 += sfmpy(r1,r3,p0):scale
    jumpr r31
  }

FUNCTION_END __hexagon_sqrtf

.global __qdsp_sqrtf ; .set __qdsp_sqrtf, __hexagon_sqrtf
.global __hexagon_fast_sqrtf ; .set __hexagon_fast_sqrtf, __hexagon_sqrtf
.global __hexagon_fast2_sqrtf ; .set __hexagon_fast2_sqrtf, __hexagon_sqrtf
