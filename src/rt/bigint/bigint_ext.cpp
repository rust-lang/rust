/* bigint_ext - external portion of large integer package
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
#include "low_primes.h"


bigint bi_0, bi_1, bi_2, bi_10, bi_m1, bi_maxint, bi_minint;


/* Forwards. */
static void print_pos( FILE* f, bigint bi );


bigint
str_to_bi( char* str )
    {
    int sign;
    bigint biR;

    sign = 1;
    if ( *str == '-' )
	{
	sign = -1;
	++str;
	}
    for ( biR = bi_0; *str >= '0' && *str <= '9'; ++str )
	biR = bi_int_add( bi_int_multiply( biR, 10 ), *str - '0' );
    if ( sign == -1 )
	biR = bi_negate( biR );
    return biR;
    }


void
bi_print( FILE* f, bigint bi )
    {
    if ( bi_is_negative( bi_copy( bi ) ) )
	{
	putc( '-', f );
	bi = bi_negate( bi );
	}
    print_pos( f, bi );
    }


bigint
bi_scan( FILE* f )
    {
    int sign;
    int c;
    bigint biR;

    sign = 1;
    c = getc( f );
    if ( c == '-' )
	sign = -1;
    else
	ungetc( c, f );

    biR = bi_0;
    for (;;)
	{
	c = getc( f );
	if ( c < '0' || c > '9' )
	    break;
	biR = bi_int_add( bi_int_multiply( biR, 10 ), c - '0' );
	}

    if ( sign == -1 )
	biR = bi_negate( biR );
    return biR;
    }


static void
print_pos( FILE* f, bigint bi )
    {
    if ( bi_compare( bi_copy( bi ), bi_10 ) >= 0 )
	print_pos( f, bi_int_divide( bi_copy( bi ), 10 ) );
    putc( bi_int_mod( bi, 10 ) + '0', f );
    }


int
bi_int_mod( bigint bi, int m )
    {
    int r;

    if ( m <= 0 )
	{
	(void) fprintf( stderr, "bi_int_mod: zero or negative modulus\n" );
	(void) kill( getpid(), SIGFPE );
	}
    r = bi_int_rem( bi, m );
    if ( r < 0 )
	r += m;
    return r;
    }


bigint
bi_rem( bigint bia, bigint bim )
    {
    return bi_subtract(
	bia, bi_multiply( bi_divide( bi_copy( bia ), bi_copy( bim ) ), bim ) );
    }


bigint
bi_mod( bigint bia, bigint bim )
    {
    bigint biR;

    if ( bi_compare( bi_copy( bim ), bi_0 ) <= 0 )
	{
	(void) fprintf( stderr, "bi_mod: zero or negative modulus\n" );
	(void) kill( getpid(), SIGFPE );
	}
    biR = bi_rem( bia, bi_copy( bim ) );
    if ( bi_is_negative( bi_copy( biR ) ) )
	biR = bi_add( biR, bim );
    else
	bi_free( bim );
    return biR;
    }


bigint
bi_square( bigint bi )
    {
    bigint biR;

    biR = bi_multiply( bi_copy( bi ), bi_copy( bi ) );
    bi_free( bi );
    return biR;
    }


bigint
bi_power( bigint bi, bigint biexp )
    {
    bigint biR;

    if ( bi_is_negative( bi_copy( biexp ) ) )
	{
	(void) fprintf( stderr, "bi_power: negative exponent\n" );
	(void) kill( getpid(), SIGFPE );
	}
    biR = bi_1;
    for (;;)
	{
	if ( bi_is_odd( bi_copy( biexp ) ) )
	    biR = bi_multiply( biR, bi_copy( bi ) );
	biexp = bi_half( biexp );
	if ( bi_compare( bi_copy( biexp ), bi_0 ) <= 0 )
	    break;
	bi = bi_multiply( bi_copy( bi ), bi );
	}
    bi_free( bi );
    bi_free( biexp );
    return biR;
    }


bigint
bi_factorial( bigint bi )
    {
    bigint biR;

    biR = bi_1;
    while ( bi_compare( bi_copy( bi ), bi_1 ) > 0 )
	{
	biR = bi_multiply( biR, bi_copy( bi ) );
	bi = bi_int_subtract( bi, 1 );
	}
    bi_free( bi );
    return biR;
    }


int
bi_is_even( bigint bi )
    {
    return ! bi_is_odd( bi );
    }


bigint
bi_mod_power( bigint bi, bigint biexp, bigint bim )
    {
    int invert;
    bigint biR;

    invert = 0;
    if ( bi_is_negative( bi_copy( biexp ) ) )
	{
	biexp = bi_negate( biexp );
	invert = 1;
	}

    biR = bi_1;
    for (;;)
	{
	if ( bi_is_odd( bi_copy( biexp ) ) )
	    biR = bi_mod( bi_multiply( biR, bi_copy( bi ) ), bi_copy( bim ) );
	biexp = bi_half( biexp );
	if ( bi_compare( bi_copy( biexp ), bi_0 ) <= 0 )
	    break;
	bi = bi_mod( bi_multiply( bi_copy( bi ), bi ), bi_copy( bim ) );
	}
    bi_free( bi );
    bi_free( biexp );

    if ( invert )
	biR = bi_mod_inverse( biR, bim );
    else
	bi_free( bim );
    return biR;
    }


bigint
bi_mod_inverse( bigint bi, bigint bim )
    {
    bigint gcd, mul0, mul1;

    gcd = bi_egcd( bi_copy( bim ), bi, &mul0, &mul1 );

    /* Did we get gcd == 1? */
    if ( ! bi_is_one( gcd ) )
	{
	(void) fprintf( stderr, "bi_mod_inverse: not relatively prime\n" );
	(void) kill( getpid(), SIGFPE );
	}

    bi_free( mul0 );
    return bi_mod( mul1, bim );
    }


/* Euclid's algorithm. */
bigint
bi_gcd( bigint bim, bigint bin )
    {
    bigint bit;

    bim = bi_abs( bim );
    bin = bi_abs( bin );
    while ( ! bi_is_zero( bi_copy( bin ) ) )
	{
	bit = bi_mod( bim, bi_copy( bin ) );
	bim = bin;
	bin = bit;
	}
    bi_free( bin );
    return bim;
    }


/* Extended Euclidean algorithm. */
bigint
bi_egcd( bigint bim, bigint bin, bigint* bim_mul, bigint* bin_mul )
    {
    bigint a0, b0, c0, a1, b1, c1, q, t;

    if ( bi_is_negative( bi_copy( bim ) ) )
	{
	bigint biR;

	biR = bi_egcd( bi_negate( bim ), bin, &t, bin_mul );
	*bim_mul = bi_negate( t );
	return biR;
	}
    if ( bi_is_negative( bi_copy( bin ) ) )
	{
	bigint biR;

	biR = bi_egcd( bim, bi_negate( bin ), bim_mul, &t );
	*bin_mul = bi_negate( t );
	return biR;
	}

    a0 = bi_1;  b0 = bi_0;  c0 = bim;
    a1 = bi_0;  b1 = bi_1;  c1 = bin;

    while ( ! bi_is_zero( bi_copy( c1 ) ) )
	{
	q = bi_divide( bi_copy( c0 ), bi_copy( c1 ) );
	t = a0;
	a0 = bi_copy( a1 );
	a1 = bi_subtract( t, bi_multiply( bi_copy( q ), a1 ) );
	t = b0;
	b0 = bi_copy( b1 );
	b1 = bi_subtract( t, bi_multiply( bi_copy( q ), b1 ) );
	t = c0;
	c0 = bi_copy( c1 );
	c1 = bi_subtract( t, bi_multiply( bi_copy( q ), c1 ) );
	bi_free( q );
	}

    bi_free( a1 );
    bi_free( b1 );
    bi_free( c1 );
    *bim_mul = a0;
    *bin_mul = b0;
    return c0;
    }


bigint
bi_lcm( bigint bia, bigint bib )
    {
    bigint biR;

    biR = bi_divide(
	bi_multiply( bi_copy( bia ), bi_copy( bib ) ),
	bi_gcd( bi_copy( bia ), bi_copy( bib ) ) );
    bi_free( bia );
    bi_free( bib );
    return biR;
    }


/* The Jacobi symbol. */
bigint
bi_jacobi( bigint bia, bigint bib )
    {
    bigint biR;

    if ( bi_is_even( bi_copy( bib ) ) )
	{
	(void) fprintf( stderr, "bi_jacobi: don't know how to compute Jacobi(n, even)\n" );
	(void) kill( getpid(), SIGFPE );
	}

    if ( bi_compare( bi_copy( bia ), bi_copy( bib ) ) >= 0 )
	return bi_jacobi( bi_mod( bia, bi_copy( bib ) ), bib );

    if ( bi_is_zero( bi_copy( bia ) ) || bi_is_one( bi_copy( bia ) ) )
	{
	bi_free( bib );
	return bia;
	}

    if ( bi_compare( bi_copy( bia ), bi_2 ) == 0 )
	{
	bi_free( bia );
	switch ( bi_int_mod( bib, 8 ) )
	    {
	    case 1: case 7:
	    return bi_1;
	    case 3: case 5:
	    return bi_m1;
	    }
	}

    if ( bi_is_even( bi_copy( bia ) ) )
	{
	biR = bi_multiply(
	    bi_jacobi( bi_2, bi_copy( bib ) ),
	    bi_jacobi( bi_half( bia ), bi_copy( bib ) ) );
	bi_free( bib );
	return biR;
	}

    if ( bi_int_mod( bi_copy( bia ), 4 ) == 3 &&
         bi_int_mod( bi_copy( bib ), 4 ) == 3 )
	return bi_negate( bi_jacobi( bib, bia ) );
    else
	return bi_jacobi( bib, bia );
    }


/* Probabalistic prime checking. */
int
bi_is_probable_prime( bigint bi, int certainty )
    {
    int i, p;
    bigint bim1;

    /* First do trial division by a list of small primes.  This eliminates
    ** many candidates.
    */
    for ( i = 0; i < sizeof(low_primes)/sizeof(*low_primes); ++i )
	{
	p = low_primes[i];
	switch ( bi_compare( int_to_bi( p ), bi_copy( bi ) ) )
	    {
	    case 0:
	    bi_free( bi );
	    return 1;
	    case 1:
	    bi_free( bi );
	    return 0;
	    }
	if ( bi_int_mod( bi_copy( bi ), p ) == 0 )
	    {
	    bi_free( bi );
	    return 0;
	    }
	}

    /* Now do the probabilistic tests. */
    bim1 = bi_int_subtract( bi_copy( bi ), 1 );
    for ( i = 0; i < certainty; ++i )
	{
	bigint a, j, jac;

	/* Pick random test number. */
	a = bi_random( bi_copy( bi ) );

	/* Decide whether to run the Fermat test or the Solovay-Strassen
	** test.  The Fermat test is fast but lets some composite numbers
	** through.  Solovay-Strassen runs slower but is more certain.
	** So the compromise here is we run the Fermat test a couple of
	** times to quickly reject most composite numbers, and then do
	** the rest of the iterations with Solovay-Strassen so nothing
	** slips through.
	*/
	if ( i < 2 && certainty >= 5 )
	    {
	    /* Fermat test.  Note that this is not state of the art.  There's a
	    ** class of numbers called Carmichael numbers which are composite
	    ** but look prime to this test - it lets them slip through no
	    ** matter how many reps you run.  However, it's nice and fast so
	    ** we run it anyway to help quickly reject most of the composites.
	    */
	    if ( ! bi_is_one( bi_mod_power( bi_copy( a ), bi_copy( bim1 ), bi_copy( bi ) ) ) )
		{
		bi_free( bi );
		bi_free( bim1 );
		bi_free( a );
		return 0;
		}
	    }
	else
	    {
	    /* GCD test.  This rarely hits, but we need it for Solovay-Strassen. */
	    if ( ! bi_is_one( bi_gcd( bi_copy( bi ), bi_copy( a ) ) ) )
		{
		bi_free( bi );
		bi_free( bim1 );
		bi_free( a );
		return 0;
		}

	    /* Solovay-Strassen test.  First compute pseudo Jacobi. */
	    j = bi_mod_power(
		    bi_copy( a ), bi_half( bi_copy( bim1 ) ), bi_copy( bi ) );
	    if ( bi_compare( bi_copy( j ), bi_copy( bim1 ) ) == 0 )
		{
		bi_free( j );
		j = bi_m1;
		}

	    /* Now compute real Jacobi. */
	    jac = bi_jacobi( bi_copy( a ), bi_copy( bi ) );

	    /* If they're not equal, the number is definitely composite. */
	    if ( bi_compare( j, jac ) != 0 )
		{
		bi_free( bi );
		bi_free( bim1 );
		bi_free( a );
		return 0;
		}
	    }

	bi_free( a );
	}

    bi_free( bim1 );

    bi_free( bi );
    return 1;
    }


bigint
bi_generate_prime( int bits, int certainty )
    {
    bigint bimo2, bip;
    int i, inc = 0;

    bimo2 = bi_power( bi_2, int_to_bi( bits - 1 ) );
    for (;;)
	{
	bip = bi_add( bi_random( bi_copy( bimo2 ) ), bi_copy( bimo2 ) );
	/* By shoving the candidate numbers up to the next highest multiple
	** of six plus or minus one, we pre-eliminate all multiples of
	** two and/or three.
	*/
	switch ( bi_int_mod( bi_copy( bip ), 6 ) )
	    {
	    case 0: inc = 4; bip = bi_int_add( bip, 1 ); break;
	    case 1: inc = 4;                             break;
	    case 2: inc = 2; bip = bi_int_add( bip, 3 ); break;
	    case 3: inc = 2; bip = bi_int_add( bip, 2 ); break;
	    case 4: inc = 2; bip = bi_int_add( bip, 1 ); break;
	    case 5: inc = 2;                             break;
	    }
	/* Starting from the generated random number, check a bunch of
	** numbers in sequence.  This is just to avoid calls to bi_random(),
	** which is more expensive than a simple add.
	*/
	for ( i = 0; i < 1000; ++i )	/* arbitrary */
	    {
	    if ( bi_is_probable_prime( bi_copy( bip ), certainty ) )
		{
		bi_free( bimo2 );
		return bip;
		}
	    bip = bi_int_add( bip, inc );
	    inc = 6 - inc;
	    }
	/* We ran through the whole sequence and didn't find a prime.
	** Shrug, just try a different random starting point.
	*/
	bi_free( bip );
	}
    }
