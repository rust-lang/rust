/*
------------------------------------------------------------------------------
rand.h: definitions for a random number generator
By Bob Jenkins, 1996, Public Domain
MODIFIED:
  960327: Creation (addition of randinit, really)
  970719: use context, not global variables, for internal state
  980324: renamed seed to flag
  980605: recommend RANDSIZL=4 for noncryptography.
  010626: note this is public domain
------------------------------------------------------------------------------
*/
#ifndef STANDARD
#include "standard.h"
#endif

#ifndef RAND
#define RAND
#define RANDSIZL   (8)  /* I recommend 8 for crypto, 4 for simulations */
#define RANDSIZ    (1<<RANDSIZL)

/* context of random number generator */
struct randctx
{
  ub4 randcnt;
  ub4 randrsl[RANDSIZ];
  ub4 randmem[RANDSIZ];
  ub4 randa;
  ub4 randb;
  ub4 randc;
};
typedef  struct randctx  randctx;

/*
------------------------------------------------------------------------------
 If (flag==TRUE), then use the contents of randrsl[0..RANDSIZ-1] as the seed.
------------------------------------------------------------------------------
*/
void randinit(randctx *r, word flag);

void isaac(randctx *r);


/*
------------------------------------------------------------------------------
 Call isaac_rand(/o_ randctx *r _o/) to retrieve a single 32-bit random value
------------------------------------------------------------------------------
*/
#define isaac_rand(r) \
   (!(r)->randcnt-- ? \
     (isaac(r), (r)->randcnt=RANDSIZ-1, (r)->randrsl[(r)->randcnt]) : \
     (r)->randrsl[(r)->randcnt])

#endif  /* RAND */


