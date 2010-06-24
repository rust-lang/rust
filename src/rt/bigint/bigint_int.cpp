/* bigint - internal portion of large integer package
**
** Copyright © 2000 by Jef Poskanzer <jef@mail.acme.com>.
** All rights reserved.
**
** Redistribution and use in source and binary forms, with or without
** modification, are permitted provided that the following conditions
** are met:
** 1. Redistributions of source code must retain the above copyright
**    notice, this list of conditions and the following disclaimer.
** 2. Redistributions in binary form must reproduce the above copyright
**    notice, this list of conditions and the following disclaimer in the
**    documentation and/or other materials provided with the distribution.
**
** THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
** ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
** IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
** ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
** FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
** DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
** OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
** HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
** LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
** OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
** SUCH DAMAGE.
*/

#include <sys/types.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

#include "bigint.h"

#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))

/* MAXINT and MININT extracted from <values.h>, which gives a warning
** message if included.
*/
#define BITSPERBYTE 8
#define BITS(type)  (BITSPERBYTE * (int)sizeof(type))
#define INTBITS     BITS(int)
#define MININT      (1 << (INTBITS - 1))
#define MAXINT      (~MININT)


/* The package represents arbitrary-precision integers as a sign and a sum
** of components multiplied by successive powers of the basic radix, i.e.:
**
**   sign * ( comp0 + comp1 * radix + comp2 * radix^2 + comp3 * radix^3 )
**
** To make good use of the computer's word size, the radix is chosen
** to be a power of two.  It could be chosen to be the full word size,
** however this would require a lot of finagling in the middle of the
** algorithms to get the inter-word overflows right.  That would slow things
** down.  Instead, the radix is chosen to be *half* the actual word size.
** With just a little care, this means the words can hold all intermediate
** values, and the overflows can be handled all at once at the end, in a
** normalization step.  This simplifies the coding enormously, and is probably
** somewhat faster to run.  The cost is that numbers use twice as much
** storage as they would with the most efficient representation, but storage
** is cheap.
**
** A few more notes on the representation:
**
**  - The sign is always 1 or -1, never 0.  The number 0 is represented
**    with a sign of 1.
**  - The components are signed numbers, to allow for negative intermediate
**    values.  After normalization, all components are >= 0 and the sign is
**    updated.
*/

/* Type definition for bigints. */
typedef int64_t comp;	/* should be the largest signed int type you have */
struct _real_bigint {
    int refs;
    struct _real_bigint* next;
    int num_comps, max_comps;
    int sign;
    comp* comps;
    };
typedef struct _real_bigint* real_bigint;


#undef DUMP


#define PERMANENT 123456789

static comp bi_radix, bi_radix_o2;
static int bi_radix_sqrt, bi_comp_bits;

static real_bigint active_list, free_list;
static int active_count, free_count;
static int check_level;


/* Forwards. */
static bigint regular_multiply( real_bigint bia, real_bigint bib );
static bigint multi_divide( bigint binumer, real_bigint bidenom );
static bigint multi_divide2( bigint binumer, real_bigint bidenom );
static void more_comps( real_bigint bi, int n );
static real_bigint alloc( int num_comps );
static real_bigint clone( real_bigint bi );
static void normalize( real_bigint bi );
static void check( real_bigint bi );
static void double_check( void );
static void triple_check( void );
#ifdef DUMP
static void dump( char* str, bigint bi );
#endif /* DUMP */
static int csqrt( comp c );
static int cbits( comp c );


void
bi_initialize( void )
    {
    /* Set the radix.  This does not actually have to be a power of
    ** two, that's just the most efficient value.  It does have to
    ** be even for bi_half() to work.
    */
    bi_radix = 1;
    bi_radix <<= BITS(comp) / 2 - 1;

    /* Halve the radix.  Only used by bi_half(). */
    bi_radix_o2 = bi_radix >> 1;

    /* Take the square root of the radix.  Only used by bi_divide(). */
    bi_radix_sqrt = csqrt( bi_radix );

    /* Figure out how many bits in a component.  Only used by bi_bits(). */
    bi_comp_bits = cbits( bi_radix - 1 );

    /* Init various globals. */
    active_list = (real_bigint) 0;
    active_count = 0;
    free_list = (real_bigint) 0;
    free_count = 0;

    /* This can be 0 through 3. */
    check_level = 3;

    /* Set up some convenient bigints. */
    bi_0 = int_to_bi( 0 ); bi_permanent( bi_0 );
    bi_1 = int_to_bi( 1 ); bi_permanent( bi_1 );
    bi_2 = int_to_bi( 2 ); bi_permanent( bi_2 );
    bi_10 = int_to_bi( 10 ); bi_permanent( bi_10 );
    bi_m1 = int_to_bi( -1 ); bi_permanent( bi_m1 );
    bi_maxint = int_to_bi( MAXINT ); bi_permanent( bi_maxint );
    bi_minint = int_to_bi( MININT ); bi_permanent( bi_minint );
    }


void
bi_terminate( void )
    {
    real_bigint p, pn;

    bi_depermanent( bi_0 ); bi_free( bi_0 );
    bi_depermanent( bi_1 ); bi_free( bi_1 );
    bi_depermanent( bi_2 ); bi_free( bi_2 );
    bi_depermanent( bi_10 ); bi_free( bi_10 );
    bi_depermanent( bi_m1 ); bi_free( bi_m1 );
    bi_depermanent( bi_maxint ); bi_free( bi_maxint );
    bi_depermanent( bi_minint ); bi_free( bi_minint );

    if ( active_count != 0 )
	(void) fprintf(
	    stderr, "bi_terminate: there were %d un-freed bigints\n",
	    active_count );
    if ( check_level >= 2 )
	double_check();
    if ( check_level >= 3 )
	{
	triple_check();
	for ( p = active_list; p != (bigint) 0; p = pn )
	    {
	    pn = p->next;
	    free( p->comps );
	    free( p );
	    }
	}
    for ( p = free_list; p != (bigint) 0; p = pn )
	{
	pn = p->next;
	free( p->comps );
	free( p );
	}
    }


void
bi_no_check( void )
    {
    check_level = 0;
    }


bigint
bi_copy( bigint obi )
    {
    real_bigint bi = (real_bigint) obi;

    check( bi );
    if ( bi->refs != PERMANENT )
	++bi->refs;
    return bi;
    }


void
bi_permanent( bigint obi )
    {
    real_bigint bi = (real_bigint) obi;

    check( bi );
    if ( check_level >= 1 && bi->refs != 1 )
	{
	(void) fprintf( stderr, "bi_permanent: refs was not 1\n" );
	(void) kill( getpid(), SIGFPE );
	}
    bi->refs = PERMANENT;
    }


void
bi_depermanent( bigint obi )
    {
    real_bigint bi = (real_bigint) obi;

    check( bi );
    if ( check_level >= 1 && bi->refs != PERMANENT )
	{
	(void) fprintf( stderr, "bi_depermanent: bigint was not permanent\n" );
	(void) kill( getpid(), SIGFPE );
	}
    bi->refs = 1;
    }


void
bi_free( bigint obi )
    {
    real_bigint bi = (real_bigint) obi;

    check( bi );
    if ( bi->refs == PERMANENT )
	return;
    --bi->refs;
    if ( bi->refs > 0 )
	return;
    if ( check_level >= 3 )
	{
	/* The active list only gets maintained at check levels 3 or higher. */
	real_bigint* nextP;
	for ( nextP = &active_list; *nextP != (real_bigint) 0; nextP = &((*nextP)->next) )
	    if ( *nextP == bi )
		{
		*nextP = bi->next;
		break;
		}
	}
    --active_count;
    bi->next = free_list;
    free_list = bi;
    ++free_count;
    if ( check_level >= 1 && active_count < 0 )
	{
	(void) fprintf( stderr,
	    "bi_free: active_count went negative - double-freed bigint?\n" );
	(void) kill( getpid(), SIGFPE );
	}
    }


int
bi_compare( bigint obia, bigint obib )
    {
    real_bigint bia = (real_bigint) obia;
    real_bigint bib = (real_bigint) obib;
    int r, c;

    check( bia );
    check( bib );

    /* First check for pointer equality. */
    if ( bia == bib )
	r = 0;
    else
	{
	/* Compare signs. */
	if ( bia->sign > bib->sign )
	    r = 1;
	else if ( bia->sign < bib->sign )
	    r = -1;
	/* Signs are the same.  Check the number of components. */
	else if ( bia->num_comps > bib->num_comps )
	    r = bia->sign;
	else if ( bia->num_comps < bib->num_comps )
	    r = -bia->sign;
	else
	    {
	    /* Same number of components.  Compare starting from the high end
	    ** and working down.
	    */
	    r = 0;	/* if we complete the loop, the numbers are equal */
	    for ( c = bia->num_comps - 1; c >= 0; --c )
		{
		if ( bia->comps[c] > bib->comps[c] )
		    { r = bia->sign; break; }
		else if ( bia->comps[c] < bib->comps[c] )
		    { r = -bia->sign; break; }
		}
	    }
	}

    bi_free( bia );
    bi_free( bib );
    return r;
    }


bigint
int_to_bi( int i )
    {
    real_bigint biR;

    biR = alloc( 1 );
    biR->sign = 1;
    biR->comps[0] = i;
    normalize( biR );
    check( biR );
    return biR;
    }


int
bi_to_int( bigint obi )
    {
    real_bigint bi = (real_bigint) obi;
    comp v, m;
    int c, r;

    check( bi );
    if ( bi_compare( bi_copy( bi ), bi_maxint ) > 0 ||
	 bi_compare( bi_copy( bi ), bi_minint ) < 0 )
	{
	(void) fprintf( stderr, "bi_to_int: overflow\n" );
	(void) kill( getpid(), SIGFPE );
	}
    v = 0;
    m = 1;
    for ( c = 0; c < bi->num_comps; ++c )
	{
	v += bi->comps[c] * m;
	m *= bi_radix;
	}
    r = (int) ( bi->sign * v );
    bi_free( bi );
    return r;
    }


bigint
bi_int_add( bigint obi, int i )
    {
    real_bigint bi = (real_bigint) obi;
    real_bigint biR;

    check( bi );
    biR = clone( bi );
    if ( biR->sign == 1 )
	biR->comps[0] += i;
    else
	biR->comps[0] -= i;
    normalize( biR );
    check( biR );
    return biR;
    }


bigint
bi_int_subtract( bigint obi, int i )
    {
    real_bigint bi = (real_bigint) obi;
    real_bigint biR;

    check( bi );
    biR = clone( bi );
    if ( biR->sign == 1 )
	biR->comps[0] -= i;
    else
	biR->comps[0] += i;
    normalize( biR );
    check( biR );
    return biR;
    }


bigint
bi_int_multiply( bigint obi, int i )
    {
    real_bigint bi = (real_bigint) obi;
    real_bigint biR;
    int c;

    check( bi );
    biR = clone( bi );
    if ( i < 0 )
	{
	i = -i;
	biR->sign = -biR->sign;
	}
    for ( c = 0; c < biR->num_comps; ++c )
	biR->comps[c] *= i;
    normalize( biR );
    check( biR );
    return biR;
    }


bigint
bi_int_divide( bigint obinumer, int denom )
    {
    real_bigint binumer = (real_bigint) obinumer;
    real_bigint biR;
    int c;
    comp r;

    check( binumer );
    if ( denom == 0 )
	{
	(void) fprintf( stderr, "bi_int_divide: divide by zero\n" );
	(void) kill( getpid(), SIGFPE );
	}
    biR = clone( binumer );
    if ( denom < 0 )
	{
	denom = -denom;
	biR->sign = -biR->sign;
	}
    r = 0;
    for ( c = biR->num_comps - 1; c >= 0; --c )
	{
	r = r * bi_radix + biR->comps[c];
	biR->comps[c] = r / denom;
	r = r % denom;
	}
    normalize( biR );
    check( biR );
    return biR;
    }


int
bi_int_rem( bigint obi, int m )
    {
    real_bigint bi = (real_bigint) obi;
    comp rad_r, r;
    int  c;

    check( bi );
    if ( m == 0 )
	{
	(void) fprintf( stderr, "bi_int_rem: divide by zero\n" );
	(void) kill( getpid(), SIGFPE );
	}
    if ( m < 0 )
	m = -m;
    rad_r = 1;
    r = 0;
    for ( c = 0; c < bi->num_comps; ++c )
	{
	r = ( r + bi->comps[c] * rad_r ) % m;
	rad_r = ( rad_r * bi_radix ) % m;
	}
    if ( bi->sign < 1 )
	r = -r;
    bi_free( bi );
    return (int) r;
    }


bigint
bi_add( bigint obia, bigint obib )
    {
    real_bigint bia = (real_bigint) obia;
    real_bigint bib = (real_bigint) obib;
    real_bigint biR;
    int c;

    check( bia );
    check( bib );
    biR = clone( bia );
    more_comps( biR, max( biR->num_comps, bib->num_comps ) );
    for ( c = 0; c < bib->num_comps; ++c )
	if ( biR->sign == bib->sign )
	    biR->comps[c] += bib->comps[c];
	else
	    biR->comps[c] -= bib->comps[c];
    bi_free( bib );
    normalize( biR );
    check( biR );
    return biR;
    }


bigint
bi_subtract( bigint obia, bigint obib )
    {
    real_bigint bia = (real_bigint) obia;
    real_bigint bib = (real_bigint) obib;
    real_bigint biR;
    int c;

    check( bia );
    check( bib );
    biR = clone( bia );
    more_comps( biR, max( biR->num_comps, bib->num_comps ) );
    for ( c = 0; c < bib->num_comps; ++c )
	if ( biR->sign == bib->sign )
	    biR->comps[c] -= bib->comps[c];
	else
	    biR->comps[c] += bib->comps[c];
    bi_free( bib );
    normalize( biR );
    check( biR );
    return biR;
    }


/* Karatsuba multiplication.  This is supposedly O(n^1.59), better than
** regular multiplication for large n.  The define below sets the crossover
** point - below that we use regular multiplication, above it we
** use Karatsuba.  Note that Karatsuba is a recursive algorithm, so
** all Karatsuba calls involve regular multiplications as the base
** steps.
*/
#define KARATSUBA_THRESH 12
bigint
bi_multiply( bigint obia, bigint obib )
    {
    real_bigint bia = (real_bigint) obia;
    real_bigint bib = (real_bigint) obib;

    check( bia );
    check( bib );
    if ( min( bia->num_comps, bib->num_comps ) < KARATSUBA_THRESH )
	return regular_multiply( bia, bib );
    else
	{
	/* The factors are large enough that Karatsuba multiplication
	** is a win.  The basic idea here is you break each factor up
	** into two parts, like so:
	**     i * r^n + j        k * r^n + l
	** r is the radix we're representing numbers with, so this
	** breaking up just means shuffling components around, no
	** math required.  With regular multiplication the product
	** would be:
	**     ik * r^(n*2) + ( il + jk ) * r^n + jl
	** That's four sub-multiplies and one addition, not counting the
	** radix-shifting.  With Karatsuba, you instead do:
	**     ik * r^(n*2) + ( (i+j)(k+l) - ik - jl ) * r^n  + jl
	** This is only three sub-multiplies.  The number of adds
	** (and subtracts) increases to four, but those run in linear time
	** so they are cheap.  The sub-multiplies are accomplished by
	** recursive calls, eventually reducing to regular multiplication.
	*/
	int n, c;
	real_bigint bi_i, bi_j, bi_k, bi_l;
	real_bigint bi_ik, bi_mid, bi_jl;

	n = ( max( bia->num_comps, bib->num_comps ) + 1 ) / 2;
	bi_i = alloc( n );
	bi_j = alloc( n );
	bi_k = alloc( n );
	bi_l = alloc( n );
	for ( c = 0; c < n; ++c )
	    {
	    if ( c + n < bia->num_comps )
		bi_i->comps[c] = bia->comps[c + n];
	    else
		bi_i->comps[c] = 0;
	    if ( c < bia->num_comps )
		bi_j->comps[c] = bia->comps[c];
	    else
		bi_j->comps[c] = 0;
	    if ( c + n < bib->num_comps )
		bi_k->comps[c] = bib->comps[c + n];
	    else
		bi_k->comps[c] = 0;
	    if ( c < bib->num_comps )
		bi_l->comps[c] = bib->comps[c];
	    else
		bi_l->comps[c] = 0;
	    }
	bi_i->sign = bi_j->sign = bi_k->sign = bi_l->sign = 1;
	normalize( bi_i );
	normalize( bi_j );
	normalize( bi_k );
	normalize( bi_l );
	bi_ik = bi_multiply( bi_copy( bi_i ), bi_copy( bi_k ) );
	bi_jl = bi_multiply( bi_copy( bi_j ), bi_copy( bi_l ) );
	bi_mid = bi_subtract(
	    bi_subtract(
		bi_multiply( bi_add( bi_i, bi_j ), bi_add( bi_k, bi_l ) ),
		bi_copy( bi_ik ) ),
	    bi_copy( bi_jl ) );
	more_comps(
	    bi_jl, max( bi_mid->num_comps + n, bi_ik->num_comps + n * 2 ) );
	for ( c = 0; c < bi_mid->num_comps; ++c )
	    bi_jl->comps[c + n] += bi_mid->comps[c];
	for ( c = 0; c < bi_ik->num_comps; ++c )
	    bi_jl->comps[c + n * 2] += bi_ik->comps[c];
	bi_free( bi_ik );
	bi_free( bi_mid );
	bi_jl->sign = bia->sign * bib->sign;
	bi_free( bia );
	bi_free( bib );
	normalize( bi_jl );
	check( bi_jl );
	return bi_jl;
	}
    }


/* Regular O(n^2) multiplication. */
static bigint
regular_multiply( real_bigint bia, real_bigint bib )
    {
    real_bigint biR;
    int new_comps, c1, c2;

    check( bia );
    check( bib );
    biR = clone( bi_0 );
    new_comps = bia->num_comps + bib->num_comps;
    more_comps( biR, new_comps );
    for ( c1 = 0; c1 < bia->num_comps; ++c1 )
	{
	for ( c2 = 0; c2 < bib->num_comps; ++c2 )
	    biR->comps[c1 + c2] += bia->comps[c1] * bib->comps[c2];
	/* Normalize after each inner loop to avoid overflowing any
	** components.  But be sure to reset biR's components count,
	** in case a previous normalization lowered it.
	*/
	biR->num_comps = new_comps;
	normalize( biR );
	}
    check( biR );
    if ( ! bi_is_zero( bi_copy( biR ) ) )
	biR->sign = bia->sign * bib->sign;
    bi_free( bia );
    bi_free( bib );
    return biR;
    }


/* The following three routines implement a multi-precision divide method
** that I haven't seen used anywhere else.  It is not quite as fast as
** the standard divide method, but it is a lot simpler.  In fact it's
** about as simple as the binary shift-and-subtract method, which goes
** about five times slower than this.
**
** The method assumes you already have multi-precision multiply and subtract
** routines, and also a multi-by-single precision divide routine.  The latter
** is used to generate approximations, which are then checked and corrected
** using the former.  The result converges to the correct value by about
** 16 bits per loop.
*/

/* Public routine to divide two arbitrary numbers. */
bigint
bi_divide( bigint binumer, bigint obidenom )
    {
    real_bigint bidenom = (real_bigint) obidenom;
    int sign;
    bigint biquotient;

    /* Check signs and trivial cases. */
    sign = 1;
    switch ( bi_compare( bi_copy( bidenom ), bi_0 ) )
	{
	case 0:
	(void) fprintf( stderr, "bi_divide: divide by zero\n" );
	(void) kill( getpid(), SIGFPE );
	case -1:
	sign *= -1;
	bidenom = bi_negate( bidenom );
	break;
	}
    switch ( bi_compare( bi_copy( binumer ), bi_0 ) )
	{
	case 0:
	bi_free( binumer );
	bi_free( bidenom );
	return bi_0;
	case -1:
	sign *= -1;
	binumer = bi_negate( binumer );
	break;
	}
    switch ( bi_compare( bi_copy( binumer ), bi_copy( bidenom ) ) )
	{
	case -1:
	bi_free( binumer );
	bi_free( bidenom );
	return bi_0;
	case 0:
	bi_free( binumer );
	bi_free( bidenom );
	if ( sign == 1 )
	    return bi_1;
	else
	    return bi_m1;
	}

    /* Is the denominator small enough to do an int divide? */
    if ( bidenom->num_comps == 1 )
	{
	/* Win! */
	biquotient = bi_int_divide( binumer, bidenom->comps[0] );
	bi_free( bidenom );
	}
    else
	{
	/* No, we have to do a full multi-by-multi divide. */
	biquotient = multi_divide( binumer, bidenom );
	}

    if ( sign == -1 )
	biquotient = bi_negate( biquotient );
    return biquotient;
    }


/* Divide two multi-precision positive numbers. */
static bigint
multi_divide( bigint binumer, real_bigint bidenom )
    {
    /* We use a successive approximation method that is kind of like a
    ** continued fraction.  The basic approximation is to do an int divide
    ** by the high-order component of the denominator.  Then we correct
    ** based on the remainder from that.
    **
    ** However, if the high-order component is too small, this doesn't
    ** work well.  In particular, if the high-order component is 1 it
    ** doesn't work at all.  Easily fixed, though - if the component
    ** is too small, increase it!
    */
    if ( bidenom->comps[bidenom->num_comps-1] < bi_radix_sqrt )
	{
	/* We use the square root of the radix as the threshhold here
	** because that's the largest value guaranteed to not make the
	** high-order component overflow and become too small again.
	**
	** We increase binumer along with bidenom to keep the end result
	** the same.
	*/
	binumer = bi_int_multiply( binumer, bi_radix_sqrt );
	bidenom = bi_int_multiply( bidenom, bi_radix_sqrt );
	}

    /* Now start the recursion. */
    return multi_divide2( binumer, bidenom );
    }


/* Divide two multi-precision positive conditioned numbers. */
static bigint
multi_divide2( bigint binumer, real_bigint bidenom )
    {
    real_bigint biapprox;
    bigint birem, biquotient;
    int c, o;

    /* Figure out the approximate quotient.   Since we're dividing by only
    ** the top component of the denominator, which is less than or equal to
    ** the full denominator, the result is guaranteed to be greater than or
    ** equal to the correct quotient.
    */
    o = bidenom->num_comps - 1;
    biapprox = bi_int_divide( bi_copy( binumer ), bidenom->comps[o] );
    /* And downshift the result to get the approximate quotient. */
    for ( c = o; c < biapprox->num_comps; ++c )
	biapprox->comps[c - o] = biapprox->comps[c];
    biapprox->num_comps -= o;

    /* Find the remainder from the approximate quotient. */
    birem = bi_subtract(
	bi_multiply( bi_copy( biapprox ), bi_copy( bidenom ) ), binumer );

    /* If the remainder is negative, zero, or in fact any value less
    ** than bidenom, then we have the correct quotient and we're done.
    */
    if ( bi_compare( bi_copy( birem ), bi_copy( bidenom ) ) < 0 )
	{
	biquotient = biapprox;
	bi_free( birem );
	bi_free( bidenom );
	}
    else
	{
	/* The real quotient is now biapprox - birem / bidenom.  We still
	** have to do a divide.  However, birem is smaller than binumer,
	** so the next divide will go faster.  We do the divide by
	** recursion.  Since this is tail-recursion or close to it, we
	** could probably re-arrange things and make it a non-recursive
	** loop, but the overhead of recursion is small and the bookkeeping
	** is simpler this way.
	**
	** Note that since the sub-divide uses the same denominator, it
	** doesn't have to adjust the values again - the high-order component
	** will still be good.
	*/
	biquotient = bi_subtract( biapprox, multi_divide2( birem, bidenom ) );
	}

    return biquotient;
    }


/* Binary division - about five times slower than the above. */
bigint
bi_binary_divide( bigint binumer, bigint obidenom )
    {
    real_bigint bidenom = (real_bigint) obidenom;
    int sign;
    bigint biquotient;

    /* Check signs and trivial cases. */
    sign = 1;
    switch ( bi_compare( bi_copy( bidenom ), bi_0 ) )
	{
	case 0:
	(void) fprintf( stderr, "bi_divide: divide by zero\n" );
	(void) kill( getpid(), SIGFPE );
	case -1:
	sign *= -1;
	bidenom = bi_negate( bidenom );
	break;
	}
    switch ( bi_compare( bi_copy( binumer ), bi_0 ) )
	{
	case 0:
	bi_free( binumer );
	bi_free( bidenom );
	return bi_0;
	case -1:
	sign *= -1;
	binumer = bi_negate( binumer );
	break;
	}
    switch ( bi_compare( bi_copy( binumer ), bi_copy( bidenom ) ) )
	{
	case -1:
	bi_free( binumer );
	bi_free( bidenom );
	return bi_0;
	case 0:
	bi_free( binumer );
	bi_free( bidenom );
	if ( sign == 1 )
	    return bi_1;
	else
	    return bi_m1;
	}

    /* Is the denominator small enough to do an int divide? */
    if ( bidenom->num_comps == 1 )
	{
	/* Win! */
	biquotient = bi_int_divide( binumer, bidenom->comps[0] );
	bi_free( bidenom );
	}
    else
	{
	/* No, we have to do a full multi-by-multi divide. */
	int num_bits, den_bits, i;

	num_bits = bi_bits( bi_copy( binumer ) );
	den_bits = bi_bits( bi_copy( bidenom ) );
	bidenom = bi_multiply( bidenom, bi_power( bi_2, int_to_bi( num_bits - den_bits ) ) );
	biquotient = bi_0;
	for ( i = den_bits; i <= num_bits; ++i )
	    {
	    biquotient = bi_double( biquotient );
	    if ( bi_compare( bi_copy( binumer ), bi_copy( bidenom ) ) >= 0 )
		{
		biquotient = bi_int_add( biquotient, 1 );
		binumer = bi_subtract( binumer, bi_copy( bidenom ) );
		}
	    bidenom = bi_half( bidenom );
	    }
	bi_free( binumer );
	bi_free( bidenom );
	}

    if ( sign == -1 )
	biquotient = bi_negate( biquotient );
    return biquotient;
    }


bigint
bi_negate( bigint obi )
    {
    real_bigint bi = (real_bigint) obi;
    real_bigint biR;

    check( bi );
    biR = clone( bi );
    biR->sign = -biR->sign;
    check( biR );
    return biR;
    }


bigint
bi_abs( bigint obi )
    {
    real_bigint bi = (real_bigint) obi;
    real_bigint biR;

    check( bi );
    biR = clone( bi );
    biR->sign = 1;
    check( biR );
    return biR;
    }


bigint
bi_half( bigint obi )
    {
    real_bigint bi = (real_bigint) obi;
    real_bigint biR;
    int c;

    check( bi );
    /* This depends on the radix being even. */
    biR = clone( bi );
    for ( c = 0; c < biR->num_comps; ++c )
	{
	if ( biR->comps[c] & 1 )
	    if ( c > 0 )
		biR->comps[c - 1] += bi_radix_o2;
	biR->comps[c] = biR->comps[c] >> 1;
	}
    /* Avoid normalization. */
    if ( biR->num_comps > 1 && biR->comps[biR->num_comps-1] == 0 )
	--biR->num_comps;
    check( biR );
    return biR;
    }


bigint
bi_double( bigint obi )
    {
    real_bigint bi = (real_bigint) obi;
    real_bigint biR;
    int c;

    check( bi );
    biR = clone( bi );
    for ( c = biR->num_comps - 1; c >= 0; --c )
	{
	biR->comps[c] = biR->comps[c] << 1;
	if ( biR->comps[c] >= bi_radix )
	    {
	    if ( c + 1 >= biR->num_comps )
		more_comps( biR, biR->num_comps + 1 );
	    biR->comps[c] -= bi_radix;
	    biR->comps[c + 1] += 1;
	    }
	}
    check( biR );
    return biR;
    }


/* Find integer square root by Newton's method. */
bigint
bi_sqrt( bigint obi )
    {
    real_bigint bi = (real_bigint) obi;
    bigint biR, biR2, bidiff;

    switch ( bi_compare( bi_copy( bi ), bi_0 ) )
	{
	case -1:
	(void) fprintf( stderr, "bi_sqrt: imaginary result\n" );
	(void) kill( getpid(), SIGFPE );
	case 0:
	return bi;
	}
    if ( bi_is_one( bi_copy( bi ) ) )
	return bi;

    /* Newton's method converges reasonably fast, but it helps to have
    ** a good initial guess.  We can make a *very* good initial guess
    ** by taking the square root of the top component times the square
    ** root of the radix part.  Both of those are easy to compute.
    */
    biR = bi_int_multiply(
	bi_power( int_to_bi( bi_radix_sqrt ), int_to_bi( bi->num_comps - 1 ) ),
	csqrt( bi->comps[bi->num_comps - 1] ) );

    /* Now do the Newton loop until we have the answer. */
    for (;;)
	{
	biR2 = bi_divide( bi_copy( bi ), bi_copy( biR ) );
	bidiff = bi_subtract( bi_copy( biR ), bi_copy( biR2 ) );
	if ( bi_is_zero( bi_copy( bidiff ) ) ||
	     bi_compare( bi_copy( bidiff ), bi_m1 ) == 0 )
	    {
	    bi_free( bi );
	    bi_free( bidiff );
	    bi_free( biR2 );
	    return biR;
	    }
	if ( bi_is_one( bi_copy( bidiff ) ) )
	    {
	    bi_free( bi );
	    bi_free( bidiff );
	    bi_free( biR );
	    return biR2;
	    }
	bi_free( bidiff );
	biR = bi_half( bi_add( biR, biR2 ) );
	}
    }


int
bi_is_odd( bigint obi )
    {
    real_bigint bi = (real_bigint) obi;
    int r;

    check( bi );
    r = bi->comps[0] & 1;
    bi_free( bi );
    return r;
    }


int
bi_is_zero( bigint obi )
    {
    real_bigint bi = (real_bigint) obi;
    int r;

    check( bi );
    r = ( bi->sign == 1 && bi->num_comps == 1 && bi->comps[0] == 0 );
    bi_free( bi );
    return r;
    }


int
bi_is_one( bigint obi )
    {
    real_bigint bi = (real_bigint) obi;
    int r;

    check( bi );
    r = ( bi->sign == 1 && bi->num_comps == 1 && bi->comps[0] == 1 );
    bi_free( bi );
    return r;
    }


int
bi_is_negative( bigint obi )
    {
    real_bigint bi = (real_bigint) obi;
    int r;

    check( bi );
    r = ( bi->sign == -1 );
    bi_free( bi );
    return r;
    }


bigint
bi_random( bigint bi )
    {
    real_bigint biR;
    int c;

    biR = bi_multiply( bi_copy( bi ), bi_copy( bi ) );
    for ( c = 0; c < biR->num_comps; ++c )
	biR->comps[c] = random();
    normalize( biR );
    biR = bi_mod( biR, bi );
    return biR;
    }


int
bi_bits( bigint obi )
    {
    real_bigint bi = (real_bigint) obi;
    int bits;

    bits =
	bi_comp_bits * ( bi->num_comps - 1 ) +
	cbits( bi->comps[bi->num_comps - 1] );
    bi_free( bi );
    return bits;
    }


/* Allocate and zero more components.  Does not consume bi, of course. */
static void
more_comps( real_bigint bi, int n )
    {
    if ( n > bi->max_comps )
	{
	bi->max_comps = max( bi->max_comps * 2, n );
	bi->comps = (comp*) realloc(
	    (void*) bi->comps, bi->max_comps * sizeof(comp) );
	if ( bi->comps == (comp*) 0 )
	    {
	    (void) fprintf( stderr, "out of memory\n" );
	    exit( 1 );
	    }
	}
    for ( ; bi->num_comps < n; ++bi->num_comps )
	bi->comps[bi->num_comps] = 0;
    }


/* Make a new empty bigint.  Fills in everything except sign and the
** components.
*/
static real_bigint
alloc( int num_comps )
    {
    real_bigint biR;

    /* Can we recycle an old bigint? */
    if ( free_list != (real_bigint) 0 )
	{
	biR = free_list;
	free_list = biR->next;
	--free_count;
	if ( check_level >= 1 && biR->refs != 0 )
	    {
	    (void) fprintf( stderr, "alloc: refs was not 0\n" );
	    (void) kill( getpid(), SIGFPE );
	    }
	more_comps( biR, num_comps );
	}
    else
	{
	/* No free bigints available - create a new one. */
	biR = (real_bigint) malloc( sizeof(struct _real_bigint) );
	if ( biR == (real_bigint) 0 )
	    {
	    (void) fprintf( stderr, "out of memory\n" );
	    exit( 1 );
	    }
	biR->comps = (comp*) malloc( num_comps * sizeof(comp) );
	if ( biR->comps == (comp*) 0 )
	    {
	    (void) fprintf( stderr, "out of memory\n" );
	    exit( 1 );
	    }
	biR->max_comps = num_comps;
	}
    biR->num_comps = num_comps;
    biR->refs = 1;
    if ( check_level >= 3 )
	{
	/* The active list only gets maintained at check levels 3 or higher. */
	biR->next = active_list;
	active_list = biR;
	}
    else
	biR->next = (real_bigint) 0;
    ++active_count;
    return biR;
    }


/* Make a modifiable copy of bi.  DOES consume bi. */
static real_bigint
clone( real_bigint bi )
    {
    real_bigint biR;
    int c;

    /* Very clever optimization. */
    if ( bi->refs != PERMANENT && bi->refs == 1 )
	return bi;

    biR = alloc( bi->num_comps );
    biR->sign = bi->sign;
    for ( c = 0; c < bi->num_comps; ++c )
	biR->comps[c] = bi->comps[c];
    bi_free( bi );
    return biR;
    }


/* Put bi into normal form.  Does not consume bi, of course.
**
** Normal form is:
**  - All components >= 0 and < bi_radix.
**  - Leading 0 components removed.
**  - Sign either 1 or -1.
**  - The number zero represented by a single 0 component and a sign of 1.
*/
static void
normalize( real_bigint bi )
    {
    int c;

    /* Borrow for negative components.  Got to be careful with the math here:
    **   -9 / 10 == 0    -9 % 10 == -9
    **   -10 / 10 == -1  -10 % 10 == 0
    **   -11 / 10 == -1  -11 % 10 == -1
    */
    for ( c = 0; c < bi->num_comps - 1; ++c )
	if ( bi->comps[c] < 0 )
	    {
	    bi->comps[c+1] += bi->comps[c] / bi_radix - 1;
	    bi->comps[c] = bi->comps[c] % bi_radix;
	    if ( bi->comps[c] != 0 )
		bi->comps[c] += bi_radix;
	    else
		bi->comps[c+1] += 1;
	    }
    /* Is the top component negative? */
    if ( bi->comps[bi->num_comps - 1] < 0 )
	{
	/* Switch the sign of the number, and fix up the components. */
	bi->sign = -bi->sign;
	for ( c = 0; c < bi->num_comps - 1; ++c )
	    {
	    bi->comps[c] =  bi_radix - bi->comps[c];
	    bi->comps[c + 1] += 1;
	    }
	bi->comps[bi->num_comps - 1] = -bi->comps[bi->num_comps - 1];
	}

    /* Carry for components larger than the radix. */
    for ( c = 0; c < bi->num_comps; ++c )
	if ( bi->comps[c] >= bi_radix )
	    {
	    if ( c + 1 >= bi->num_comps )
		more_comps( bi, bi->num_comps + 1 );
	    bi->comps[c+1] += bi->comps[c] / bi_radix;
	    bi->comps[c] = bi->comps[c] % bi_radix;
	    }

    /* Trim off any leading zero components. */
    for ( ; bi->num_comps > 1 && bi->comps[bi->num_comps-1] == 0; --bi->num_comps )
	;

    /* Check for -0. */
    if ( bi->num_comps == 1 && bi->comps[0] == 0 && bi->sign == -1 )
	bi->sign = 1;
    }


static void
check( real_bigint bi )
    {
    if ( check_level == 0 )
	return;
    if ( bi->refs == 0 )
	{
	(void) fprintf( stderr, "check: zero refs in bigint\n" );
	(void) kill( getpid(), SIGFPE );
	}
    if ( bi->refs < 0 )
	{
	(void) fprintf( stderr, "check: negative refs in bigint\n" );
	(void) kill( getpid(), SIGFPE );
	}
    if ( check_level < 3 )
	{
	/* At check levels less than 3, active bigints have a zero next. */
	if ( bi->next != (real_bigint) 0 )
	    {
	    (void) fprintf(
		stderr, "check: attempt to use a bigint from the free list\n" );
	    (void) kill( getpid(), SIGFPE );
	    }
	}
    else
	{
	/* At check levels 3 or higher, active bigints must be on the active
	** list.
	*/
	real_bigint p;

	for ( p = active_list; p != (real_bigint) 0; p = p->next )
	    if ( p == bi )
		break;
	if ( p == (real_bigint) 0 )
	    {
	    (void) fprintf( stderr,
		"check: attempt to use a bigint not on the active list\n" );
	    (void) kill( getpid(), SIGFPE );
	    }
	}
    if ( check_level >= 2 )
	double_check();
    if ( check_level >= 3 )
	triple_check();
    }


static void
double_check( void )
    {
    real_bigint p;
    int c;

    for ( p = free_list, c = 0; p != (real_bigint) 0; p = p->next, ++c )
	if ( p->refs != 0 )
	    {
	    (void) fprintf( stderr,
		"double_check: found a non-zero ref on the free list\n" );
	    (void) kill( getpid(), SIGFPE );
	    }
    if ( c != free_count )
	{
	(void) fprintf( stderr,
	    "double_check: free_count is %d but the free list has %d items\n",
	    free_count, c );
	(void) kill( getpid(), SIGFPE );
	}
    }


static void
triple_check( void )
    {
    real_bigint p;
    int c;

    for ( p = active_list, c = 0; p != (real_bigint) 0; p = p->next, ++c )
	if ( p->refs == 0 )
	    {
	    (void) fprintf( stderr,
		"triple_check: found a zero ref on the active list\n" );
	    (void) kill( getpid(), SIGFPE );
	    }
    if ( c != active_count )
	{
	(void) fprintf( stderr,
	    "triple_check: active_count is %d but active_list has %d items\n",
	    free_count, c );
	(void) kill( getpid(), SIGFPE );
	}
    }


#ifdef DUMP
/* Debug routine to dump out a complete bigint.  Does not consume bi. */
static void
dump( char* str, bigint obi )
    {
    int c;
    real_bigint bi = (real_bigint) obi;

    (void) fprintf( stdout, "dump %s at 0x%08x:\n", str, (unsigned int) bi );
    (void) fprintf( stdout, "  refs: %d\n", bi->refs );
    (void) fprintf( stdout, "  next: 0x%08x\n", (unsigned int) bi->next );
    (void) fprintf( stdout, "  num_comps: %d\n", bi->num_comps );
    (void) fprintf( stdout, "  max_comps: %d\n", bi->max_comps );
    (void) fprintf( stdout, "  sign: %d\n", bi->sign );
    for ( c = bi->num_comps - 1; c >= 0; --c )
	(void) fprintf( stdout, "    comps[%d]: %11lld (0x%016llx)\n", c, (long long) bi->comps[c], (long long) bi->comps[c] );
    (void) fprintf( stdout, "  print: " );
    bi_print( stdout, bi_copy( bi ) );
    (void) fprintf( stdout, "\n" );
    }
#endif /* DUMP */


/* Trivial square-root routine so that we don't have to link in the math lib. */
static int
csqrt( comp c )
    {
    comp r, r2, diff;

    if ( c < 0 )
	{
	(void) fprintf( stderr, "csqrt: imaginary result\n" );
	(void) kill( getpid(), SIGFPE );
	}

    r = c / 2;
    for (;;)
	{
	r2 = c / r;
	diff = r - r2;
	if ( diff == 0 || diff == -1 )
	    return (int) r;
	if ( diff == 1 )
	    return (int) r2;
	r = ( r + r2 ) / 2;
	}
    }


/* Figure out how many bits are in a number. */
static int
cbits( comp c )
    {
    int b;

    for ( b = 0; c != 0; ++b )
	c >>= 1;
    return b;
    }
