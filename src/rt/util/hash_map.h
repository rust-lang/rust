// -*- c++ -*-
/**
 * A C++ wrapper around uthash.
 */

#ifndef HASH_MAP
#define HASH_MAP

#include <assert.h>
#include "../uthash/uthash.h"

template<typename K, typename V> class hash_map {
    struct map_entry {
        K key;
        V value;
        UT_hash_handle hh;
    };
    map_entry * _head;
public:
    hash_map();
    ~hash_map();

    /**
     * Associates a value with the specified key in this hash map.
     * If a mapping already exists the old value is replaced.
     *
     * returns:
     * true if the mapping was successfully created and false otherwise.
     */
    bool put(K key, V value);

    /**
     * Updates the value associated with the specified key in this hash map.
     *
     * returns:
     * true if the value was updated, or false if the key was not found.
     */
    bool set(K key, V value);

    /**
     * Gets the value associated with the specified key in this hash map.
     *
     * returns:
     * true if the value was found and updates the specified *value parameter
     * with the associated value, or false otherwise.
     */
    bool get(K key, V *value);

    /**
     * Removes a key-value pair from this hash map.
     *
     * returns:
     * true if a key-value pair exists and updates the specified
     * *key and *value parameters, or false otherwise.
     */
    bool pop(K *key, V *value);

    /**
     * Checks if the specified key exists in this hash map.
     *
     * returns:
     * true if the specified key exists in this hash map, or false otherwise.
     */
    bool contains(K key);

    /**
     * Removes the value associated with the specified key from this hash map.
     *
     * returns:
     * true if the specified key exists and updates the specified *old_value
     * parameter with the associated value, or false otherwise.
     */
    bool remove(K key, V *old_value);
    bool remove(K key);

    /**
     * Returns the number of key-value pairs in this hash map.
     */
    size_t count();

    bool is_empty() {
        return count() == 0;
    }

    /**
     * Clears all the key-value pairs in this hash map.
     *
     * returns:
     * the number of deleted key-value pairs.
     */
    size_t clear();
};

template<typename K, typename V>
hash_map<K,V>::hash_map() {
    _head = NULL;
}

template<typename K, typename V>
hash_map<K,V>::~hash_map() {
    clear();
}

template<typename K, typename V> bool
hash_map<K,V>::put(K key, V value) {
    if (contains(key)) {
        return set(key, value);
    }
    map_entry *entry = (map_entry *) malloc(sizeof(map_entry));
    entry->key = key;
    entry->value = value;
    HASH_ADD(hh, _head, key, sizeof(K), entry);
    return true;
}

template<typename K, typename V> bool
hash_map<K,V>::get(K key, V *value) {
    map_entry *entry = NULL;
    HASH_FIND(hh, _head, &key, sizeof(K), entry);
    if (entry == NULL) {
        return false;
    }
    *value = entry->value;
    return true;
}

template<typename K, typename V> bool
hash_map<K,V>::set(K key, V value) {
    map_entry *entry = NULL;
    HASH_FIND(hh, _head, &key, sizeof(K), entry);
    if (entry == NULL) {
        return false;
    }
    entry->value = value;
    return true;
}

template<typename K, typename V> bool
hash_map<K,V>::contains(K key) {
    V value;
    return get(key, &value);
}

template<typename K, typename V> bool
hash_map<K,V>::remove(K key, V *old_value) {
    map_entry *entry = NULL;
    HASH_FIND(hh, _head, &key, sizeof(K), entry);
    if (entry == NULL) {
        return false;
    }
    *old_value = entry->value;
    HASH_DEL(_head, entry);
    free(entry);
    return true;
}

template<typename K, typename V> bool
hash_map<K,V>::pop(K *key, V *value) {
    if (is_empty()) {
        return false;
    }
    map_entry *entry = _head;
    HASH_DEL(_head, entry);
    *key = entry->key;
    *value = entry->value;
    free(entry);
    return true;
}

template<typename K, typename V> bool
hash_map<K,V>::remove(K key) {
    V old_value;
    return remove(key, &old_value);
}

template<typename K, typename V> size_t
hash_map<K,V>::count() {
    return HASH_CNT(hh, _head);
}

template<typename K, typename V> size_t
hash_map<K,V>::clear() {
    size_t deleted_entries = 0;
    while (_head != NULL) {
        map_entry *entry = _head;
        HASH_DEL(_head, entry);
        free(entry);
        deleted_entries ++;
    }
    assert(count() == 0);
    return deleted_entries;
}

#endif /* HASH_MAP */
