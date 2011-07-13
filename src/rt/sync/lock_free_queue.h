#ifndef LOCK_FREE_QUEUE_H
#define LOCK_FREE_QUEUE_H

/**
 * How and why this lock free queue works:
 *
 * Adapted from the paper titled "Simple, Fast, and Practical Non-Blocking
 * and Blocking Concurrent Queue Algorithms" by Maged M. Michael,
 * Michael L. Scott.
 *
 * Safety Properties:
 *
 * 1. The linked list is always connected.
 * 2. Nodes are only inserted after the last node in the linked list.
 * 3. Nodes are only deleted from the beginning of the linked list.
 * 4. Head always points to the first node in the linked list.
 * 5. Tail always points to a node in the linked list.
 *
 *
 * 1. The linked list is always connected because the next pointer is not set
 *    to null before the node is freed, and no node is freed until deleted
 *    from the linked list.
 *
 * 2. Nodes are only inserted at the end of the linked list because they are
 *    linked through the tail pointer which always points to a node in the
 *    linked list (5) and an inserted node is only linked to a node that has
 *    a null next pointer, and the only such node is the last one (1).
 *
 * 3. Nodes are deleted from the beginning of the list because they are
 *    deleted only when they are pointed to by head which always points to the
 *    first node (4).
 *
 * 4. Head always points to the first node in the list because it only changes
 *    its value to the next node atomically. The new value of head cannot be
 *    null because if there is only one node in the list the dequeue operation
 *    returns without deleting any nodes.
 *
 * 5. Tail always points to a node in the linked list because it never lags
 *    behind head, so it can never point to a deleted node. Also, when tail
 *    changes its value it always swings to the next node in the list and it
 *    never tires to change its value if the next pointer is NULL.
 */

#include <assert.h>
template <class T>
class lock_free_queue {

    struct node_t;
    struct pointer_t {
        node_t *node;
        uint32_t count;
        pointer_t() : node(NULL), count(0) {
        }
        pointer_t(node_t *node, uint32_t count) {
            this->node = node;
            this->count = count;
        }
        bool equals(pointer_t &other) {
            return node == other.node && count == other.count;
        }
    };

    struct node_t {
        T value;
        pointer_t next;

        node_t() {
            next.node = NULL;
            next.count = 0;
        }

        node_t(pointer_t next, T value) {
            this->next = next;
            this->value = value;
        }
    };

    // Always points to the first node in the list.
    pointer_t head;

    // Always points to a node in the list, (not necessarily the last).
    pointer_t tail;

    // Compare and swap counted pointers, we can only do this if pointr_t is
    // 8 bytes or less since that the maximum size CAS can handle.
    bool compare_and_swap(pointer_t *address,
        pointer_t *oldValue,
        pointer_t newValue) {

        // FIXME this is requiring us to pass -fno-strict-aliasing to GCC
        // (possibly there are other, similar problems)
        if (sync::compare_and_swap(
                (uint64_t*) address,
                *(uint64_t*) oldValue,
                *(uint64_t*) &newValue)) {
            return true;
        }
        return false;
    }

public:
    lock_free_queue() {
        // We can only handle 64bit CAS for counted pointers, so this will
        // not work with 64bit pointers.
        assert (sizeof(pointer_t) == sizeof(uint64_t));

        // Allocate a dummy node to be used as the first node in the list.
        node_t *node = new node_t();

        // Head and tail both start out pointing to the dummy node.
        head.node = node;
        tail.node = node;
    }

    virtual ~lock_free_queue() {
        // Delete dummy node.
        delete head.node;
    }

    bool is_empty() {
        return head.node == tail.node;
    }

    virtual void enqueue(T value) {

        // Create a new node to be inserted in the linked list, and set the
        // next node to NULL.
        node_t *node = new node_t();
        node->value = value;
        node->next.node = NULL;
        pointer_t tail;

        // Keep trying until enqueue is done.
        while (true) {
            // Read the current tail which may either point to the last node
            // or to the second to last node (not sure why second to last,
            // and not any other node).
            tail = this->tail;

            // Reads the next node after the tail which will be the last node
            // if null.
            pointer_t next;
            if (tail.node != NULL) {
                next = tail.node->next;
            }

            // Loop if another thread changed the tail since we last read it.
            if (tail.equals(this->tail) == false) {
                continue;
            }

            // If next is not pointing to the last node try to swing tail to
            // the last node and loop.
            if (next.node != NULL) {
                compare_and_swap(&this->tail, &tail,
                    pointer_t(next.node, tail.count + 1));
                continue;
            }

            // Try to link node at the end of the linked list.
            if (compare_and_swap(&tail.node->next, &next,
                    pointer_t(node, next.count + 1))) {
                // Enqueueing is done.
                break;
            }
        }

        // Enqueue is done, try to swing tail to the inserted node.
        compare_and_swap(&this->tail, &tail,
            pointer_t(node, tail.count + 1));
    }

    bool dequeue(T *value) {
        pointer_t head;

        // Keep trying until dequeue is done.
        while(true) {
            head = this->head;
            pointer_t tail = this->tail;
            pointer_t next = head.node->next;

            if (head.equals(this->head) == false) {
                continue;
            }

            // If queue is empty, or if tail is falling behind.
            if (head.node == tail.node) {
                // If queue is empty.
                if (next.node == NULL) {
                    return false;
                }
                // Tail is falling behind, advance it.
                compare_and_swap(&this->tail,
                    &tail,
                    pointer_t(next.node, tail.count + 1));
            } else {
                // Read value before CAS, otherwise another
                // dequeue might advance it.
                *value = next.node->value;
                if (compare_and_swap(&this->head, &head,
                    pointer_t(next.node, head.count + 1))) {
                    break;
                }
            }
        }
        delete head.node;
        return true;
    }
};

#endif /* LOCK_FREE_QUEUE_H */
