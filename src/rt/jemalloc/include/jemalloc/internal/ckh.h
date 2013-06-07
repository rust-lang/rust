/******************************************************************************/
#ifdef JEMALLOC_H_TYPES

typedef struct ckh_s ckh_t;
typedef struct ckhc_s ckhc_t;

/* Typedefs to allow easy function pointer passing. */
typedef void ckh_hash_t (const void *, size_t[2]);
typedef bool ckh_keycomp_t (const void *, const void *);

/* Maintain counters used to get an idea of performance. */
/* #define	CKH_COUNT */
/* Print counter values in ckh_delete() (requires CKH_COUNT). */
/* #define	CKH_VERBOSE */

/*
 * There are 2^LG_CKH_BUCKET_CELLS cells in each hash table bucket.  Try to fit
 * one bucket per L1 cache line.
 */
#define LG_CKH_BUCKET_CELLS (LG_CACHELINE - LG_SIZEOF_PTR - 1)

#endif /* JEMALLOC_H_TYPES */
/******************************************************************************/
#ifdef JEMALLOC_H_STRUCTS

/* Hash table cell. */
struct ckhc_s {
    const void	*key;
    const void	*data;
};

struct ckh_s {
#ifdef CKH_COUNT
    /* Counters used to get an idea of performance. */
    uint64_t	ngrows;
    uint64_t	nshrinks;
    uint64_t	nshrinkfails;
    uint64_t	ninserts;
    uint64_t	nrelocs;
#endif

    /* Used for pseudo-random number generation. */
#define	CKH_A		1103515241
#define	CKH_C		12347
    uint32_t	prng_state;

    /* Total number of items. */
    size_t		count;

    /*
     * Minimum and current number of hash table buckets.  There are
     * 2^LG_CKH_BUCKET_CELLS cells per bucket.
     */
    unsigned	lg_minbuckets;
    unsigned	lg_curbuckets;

    /* Hash and comparison functions. */
    ckh_hash_t	*hash;
    ckh_keycomp_t	*keycomp;

    /* Hash table with 2^lg_curbuckets buckets. */
    ckhc_t		*tab;
};

#endif /* JEMALLOC_H_STRUCTS */
/******************************************************************************/
#ifdef JEMALLOC_H_EXTERNS

bool	ckh_new(ckh_t *ckh, size_t minitems, ckh_hash_t *hash,
    ckh_keycomp_t *keycomp);
void	ckh_delete(ckh_t *ckh);
size_t	ckh_count(ckh_t *ckh);
bool	ckh_iter(ckh_t *ckh, size_t *tabind, void **key, void **data);
bool	ckh_insert(ckh_t *ckh, const void *key, const void *data);
bool	ckh_remove(ckh_t *ckh, const void *searchkey, void **key,
    void **data);
bool	ckh_search(ckh_t *ckh, const void *seachkey, void **key, void **data);
void	ckh_string_hash(const void *key, size_t r_hash[2]);
bool	ckh_string_keycomp(const void *k1, const void *k2);
void	ckh_pointer_hash(const void *key, size_t r_hash[2]);
bool	ckh_pointer_keycomp(const void *k1, const void *k2);

#endif /* JEMALLOC_H_EXTERNS */
/******************************************************************************/
#ifdef JEMALLOC_H_INLINES

#endif /* JEMALLOC_H_INLINES */
/******************************************************************************/
