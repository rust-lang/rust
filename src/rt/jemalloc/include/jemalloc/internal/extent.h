/******************************************************************************/
#ifdef JEMALLOC_H_TYPES

typedef struct extent_node_s extent_node_t;

#endif /* JEMALLOC_H_TYPES */
/******************************************************************************/
#ifdef JEMALLOC_H_STRUCTS

/* Tree of extents. */
struct extent_node_s {
	/* Linkage for the size/address-ordered tree. */
	rb_node(extent_node_t)	link_szad;

	/* Linkage for the address-ordered tree. */
	rb_node(extent_node_t)	link_ad;

	/* Profile counters, used for huge objects. */
	prof_ctx_t		*prof_ctx;

	/* Pointer to the extent that this tree node is responsible for. */
	void			*addr;

	/* Total region size. */
	size_t			size;

	/* True if zero-filled; used by chunk recycling code. */
	bool			zeroed;
};
typedef rb_tree(extent_node_t) extent_tree_t;

#endif /* JEMALLOC_H_STRUCTS */
/******************************************************************************/
#ifdef JEMALLOC_H_EXTERNS

rb_proto(, extent_tree_szad_, extent_tree_t, extent_node_t)

rb_proto(, extent_tree_ad_, extent_tree_t, extent_node_t)

#endif /* JEMALLOC_H_EXTERNS */
/******************************************************************************/
#ifdef JEMALLOC_H_INLINES

#endif /* JEMALLOC_H_INLINES */
/******************************************************************************/

